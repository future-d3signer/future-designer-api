import io
import cv2
import base64
import numpy as np
from PIL import Image
from fastapi import HTTPException


class ImageUtils:
    @staticmethod
    def encode_image(image: Image.Image, format="JPEG") -> str:
        buffered = io.BytesIO()
        if isinstance(image, (list, np.ndarray)):
            image = Image.fromarray(np.uint8(image))
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def decode_image(image_base64: str) -> Image.Image:
        try:
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )
    
    @staticmethod
    def add_mask_padding(mask_image: Image.Image, padding: int = 20) -> Image.Image:
        mask_np = np.array(mask_image.convert('L'))
        kernel = np.ones((padding, padding), np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        
        return Image.fromarray(dilated_mask)