import os
import cv2
import torch
import numpy as np

from PIL import Image
from copy import deepcopy
from app.core.config import settings
from typing import Tuple, List 
from sam2.sam2_image_predictor import SAM2ImagePredictor


os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def apply_coords_torch(
    coords: torch.Tensor, original_size, image_size
) -> torch.Tensor:
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
    boxes = apply_coords_torch(boxes.reshape(-1, 2, 2), original_size, image_size)
    return boxes.reshape(-1, 4)

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def preprocess_image(image_source_np: np.array) -> torch.Tensor: 
    image = image_source_np.astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return image_tensor.to(device=DEVICE, dtype=torch.float32)

def load_sam_predictor() -> SAM2ImagePredictor: 
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
    return predictor


def _segment_furniture_with_models(
    image_pil: Image.Image,
    grounding_dino_model,
    grounding_dino_processor, 
    sam_predictor: SAM2ImagePredictor 
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]: 

    image_source_np = np.asarray(image_pil)
    inputs = grounding_dino_processor(images=image_pil, text=settings.DINO_PROMPT, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)

    results = grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.4,
        target_sizes=[image_pil.size[::-1]]
    )

    boxes = results[0]['boxes']  

    if boxes.nelement() == 0: 
        return np.array([]), torch.empty(0), torch.empty(0)

    image_tensor = preprocess_image(image_source_np) 
    sam_predictor.set_image(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    boxes.to(device=DEVICE, dtype=torch.float16)

    masks = np.array([])
    for i in range(len(boxes)):
        mask, scores, _ = sam_predictor.predict(
                point_coords = None,
                point_labels = None,
                box = boxes[i],
                multimask_output = False,
            )
        
        masks = np.array([mask]) if masks.size == 0 else np.concatenate((masks, [mask]), axis=0)
    
    return masks, scores, boxes

def extract_furniture_segments_from_image(
    image_pil: Image.Image,
    sam_predictor: SAM2ImagePredictor, 
    grounding_dino_model,
    grounding_dino_processor 
) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor]: 

    image_source_np = np.array(image_pil) 

    masks, _, boxes = _segment_furniture_with_models(
        image_pil, grounding_dino_model, grounding_dino_processor, sam_predictor
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