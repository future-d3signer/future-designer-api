import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from copy import deepcopy
from groundingdino.util.inference import predict
from groundingdino.util import box_ops
import cv2
from PIL import Image
from typing import Tuple
import groundingdino.datasets.transforms as T
from io import BytesIO
import base64
import os

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_image(image):
    """Preprocess image to match model's expected input type"""
    # Convert image to float32 first for better precision during preprocessing
    image = image.astype(np.float32) / 255.0
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # Convert to float32 and move to correct device
    return image_tensor.to(device=DEVICE, dtype=torch.float32)


def load_image(image_source: Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def load_sam_model(checkpoint_path="sam2_h.pth", device="cuda"):
    checkpoint = "/home/s464915/future-designer/experiments/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam = build_sam2(model_cfg, checkpoint)
    sam.to(device=DEVICE)
    predictor = SAM2ImagePredictor(sam)
    return predictor

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

def segment_furniture(image, grounding_dino_model, predictor):
    image_source, gd_image = load_image(image)
    
    # Get furniture detections
    boxes, logits, phrases = predict(
        model=grounding_dino_model,
        image=gd_image,
        caption="chair. sofa. table. bed.",
        box_threshold=0.4,
        text_threshold=0.4,
        device=DEVICE
    )

    # Prepare image for SAM
    image_tensor = preprocess_image(image_source)
    predictor.set_image(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    boxes.to(device=DEVICE, dtype=torch.float16)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes).to(DEVICE) * torch.tensor([W, H, W, H], dtype=torch.float16, device=DEVICE)

    transformed_boxes = apply_boxes_torch(boxes_xyxy, image_source.shape[:2], image_source.shape[1]).to(DEVICE)

    masks = np.array([])
    for i in range(len(transformed_boxes)):
        mask, scores, _ = predictor.predict(
                point_coords = None,
                point_labels = None,
                box = transformed_boxes[i],
                multimask_output = False,
            )
        
        masks = np.array([mask]) if masks.size == 0 else np.concatenate((masks, [mask]), axis=0)
    
    return masks, scores, transformed_boxes

def get_segementaion(image: Image, sam_model, grounding_dino_model):
    image_source = np.array(image)
    masks, _, boxes = segment_furniture(image, grounding_dino_model, sam_model)

    segmentation_masks = []

    for i in range(len(masks)):
        x0, y0, x1, y1 = boxes[i].int().cpu().numpy()
        mask_slice = masks[i, 0, y0:y1, x0:x1]
        inverted_mask = (1 - mask_slice).astype(np.uint8) * 255  # Invert the mask

        # Create a white canvas with the same shape as the ROI
        roi = image_source[y0:y1, x0:x1]
        white_bg = np.ones_like(roi, dtype=np.uint8) * 255

        # Fill the mask region with the original image
        mask_slice = cv2.bitwise_or(
            cv2.bitwise_and(roi, roi, mask=255 - inverted_mask),
            cv2.bitwise_and(white_bg, white_bg, mask=inverted_mask)
        )
        mask_slice = Image.fromarray(mask_slice)
        segmentation_masks.append(mask_slice)

    return segmentation_masks, masks, boxes