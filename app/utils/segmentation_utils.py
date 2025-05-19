import numpy as np
import torch
from sam2.build_sam import build_sam2 # Assuming sam2 is in PYTHONPATH or installable
from sam2.sam2_image_predictor import SAM2ImagePredictor
from copy import deepcopy
from groundingdino.util.inference import predict as dino_predict # Alias to avoid conflict if you had another 'predict'
from groundingdino.util import box_ops
import cv2
from PIL import Image
from typing import Tuple, List # Added List for type hints
import groundingdino.datasets.transforms as T
# from io import BytesIO # Not used directly here
# import base64 # Not used directly here
import os

from app.core.config import settings # Import your global settings

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1' # This is fine here or in main.py startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def apply_coords_torch(
    coords: torch.Tensor, original_size, image_size
) -> torch.Tensor:
    """
    Expects a torch tensor with length 2 in the last dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], image_size
    )
    coords = deepcopy(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def apply_boxes_torch(
    boxes: torch.Tensor, original_size, image_size
) -> torch.Tensor:
    """
    Expects a torch tensor with shape Bx4. Requires the original image
    size in (H, W) format.
    """
    boxes = apply_coords_torch(boxes.reshape(-1, 2, 2), original_size, image_size)
    return boxes.reshape(-1, 4)

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def preprocess_image(image_source_np: np.array) -> torch.Tensor: # Added type hint for input
    """Preprocess image to match model's expected input type"""
    # Convert image to float32 first for better precision during preprocessing
    # Assuming image_source_np is already HWC, RGB, 0-255 uint8
    image = image_source_np.astype(np.float32) / 255.0
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # Convert to float32 and move to correct device
    return image_tensor.to(device=DEVICE, dtype=torch.float32)


def load_dino_image_transform(image_pil: Image.Image) -> Tuple[np.array, torch.Tensor]: # Renamed and clarified input
    """
    Loads a PIL image, converts to numpy array, and applies GroundingDINO transforms.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_np = np.asarray(image_pil)
    image_transformed, _ = transform(image_pil, None) # Pass PIL image to transform
    return image_np, image_transformed

def load_sam_predictor() -> SAM2ImagePredictor: # Renamed for clarity, SAM model itself vs predictor
    """Loads the SAM2 model and returns an image predictor instance."""
    # Using paths from global settings
    sam = build_sam2(settings.SAM2_MODEL_CONFIG, settings.SAM2_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SAM2ImagePredictor(sam)
    return predictor

# apply_coords_torch, apply_boxes_torch, get_preprocess_shape remain the same

def _segment_furniture_with_models(
    image_pil: Image.Image,
    grounding_dino_model, # Pass the loaded model
    sam_predictor: SAM2ImagePredictor # Pass the loaded predictor
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]: # Added types
    """
    Internal function to perform DINO detection and SAM segmentation.
    Assumes models are already loaded and passed in.
    """
    image_source_np, gd_image_tensor = load_dino_image_transform(image_pil)

    # Get furniture detections
    boxes, logits, phrases = dino_predict( # Use aliased dino_predict
        model=grounding_dino_model,
        image=gd_image_tensor,
        caption="chair. sofa. table. bed.", # Consider making this configurable
        box_threshold=0.4, # Consider making this configurable
        text_threshold=0.4, # Consider making this configurable
        device=DEVICE
    )

    if boxes.nelement() == 0: # No boxes detected
        return np.array([]), torch.empty(0), torch.empty(0)


    # Prepare image for SAM (SAM2ImagePredictor expects a NumPy array (H, W, C) in RGB order)
    # The preprocess_image function was more for SAM v1. SAM2ImagePredictor might handle it.
    # predictor.set_image(image_source_np) # For SAM2ImagePredictor, this is how you set the image

    # According to SAM2ImagePredictor docs/examples, it typically takes a BGR numpy array.
    # If your image_source_np is RGB, convert it:
    image_tensor = preprocess_image(image_source_np) # Preprocess for SAM
    sam_predictor.set_image(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    boxes.to(device=DEVICE, dtype=torch.float16)

    # Original code used `image_tensor` which was preprocessed differently.
    # Let's stick to what SAM2ImagePredictor expects.
    # If predictor.set_image needs specific normalization not done by load_dino_image_transform for image_source_np,
    # that needs to be handled. Assuming image_source_np (0-255, uint8, RGB) is what set_image can start with.

    # Convert DINO boxes to SAM expected format
    H, W, _ = image_source_np.shape
    # DINO boxes are cxcywh normalized, convert to xyxy absolute
    boxes_xyxy_abs = box_ops.box_cxcywh_to_xyxy(boxes).to(DEVICE) * torch.tensor([W, H, W, H], dtype=torch.float16, device=DEVICE)
    # boxes_xyxy_abs = boxes_xyxy_abs.to(DEVICE, dtype=torch.float32) # ensure correct type and device
    transformed_boxes = apply_boxes_torch(boxes_xyxy_abs, image_source_np.shape[:2], image_source_np.shape[1]).to(DEVICE)
    # The original code had `transformed_boxes = apply_boxes_torch(...)`.
    # SAM2ImagePredictor's `predict` method takes boxes in the original image coordinate system.
    # So, `boxes_xyxy_abs` should be directly usable if `set_image` used the original unresized image.
    # Let's assume `apply_boxes_torch` is not needed if SAM is set with the original image.
    # If SAM was set with a resized/transformed image, then box transformation is vital.
    # Given `sam_predictor.set_image(image_bgr_np)` uses the original scale image,
    # `boxes_xyxy_abs` should be the correct input for `sam_predictor.predict(box=...)`.

    
    masks = np.array([])
    for i in range(len(transformed_boxes)):
        mask, scores, _ = sam_predictor.predict(
                point_coords = None,
                point_labels = None,
                box = transformed_boxes[i],
                multimask_output = False,
            )
        
        masks = np.array([mask]) if masks.size == 0 else np.concatenate((masks, [mask]), axis=0)
    
    return masks, scores, transformed_boxes
 # Or phrases instead of logits if needed


def extract_furniture_segments_from_image(
    image_pil: Image.Image,
    sam_predictor: SAM2ImagePredictor, # Pass the loaded predictor
    grounding_dino_model # Pass the loaded model
) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor]: # Renamed for clarity
    """
    Detects and segments furniture from a PIL image.

    Returns:
        - A list of PIL Images, each containing a segmented furniture item on a white background.
        - Raw SAM masks as a NumPy array (N, H, W).
        - Detected bounding boxes (N, 4) in xyxy format (absolute coordinates).
    """
    image_source_np = np.array(image_pil) # Original image as numpy array (RGB)

    # sam_masks_raw: (num_detected_items, H, W) boolean or float masks from SAM
    # dino_scores: scores for detected items
    # dino_boxes_xyxy: (num_detected_items, 4) tensor [x1,y1,x2,y2] absolute
    masks, _, boxes = _segment_furniture_with_models(
        image_pil, grounding_dino_model, sam_predictor
    )

    segmentation_masks = []

    for i in range(len(masks)):
        x0, y0, x1, y1 = boxes[i].int().cpu().numpy()
        mask_slice = masks[i, 0, y0:y1, x0:x1]
        inverted_mask = (1 - mask_slice).astype(np.uint8) * 255  # Invert the mask

        # Create a white canvas with the same shape as the ROI
        roi = image_source_np[y0:y1, x0:x1]
        white_bg = np.ones_like(roi, dtype=np.uint8) * 255

        # Fill the mask region with the original image
        mask_slice = cv2.bitwise_or(
            cv2.bitwise_and(roi, roi, mask=255 - inverted_mask),
            cv2.bitwise_and(white_bg, white_bg, mask=inverted_mask)
        )
        mask_slice = Image.fromarray(mask_slice)
        segmentation_masks.append(mask_slice)

    return segmentation_masks, masks, boxes