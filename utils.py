import io
import json
import base64
import numpy as np

from PIL import Image, ImageFilter
from fastapi import HTTPException
import cv2

class ImageUtils:
    @staticmethod
    def encode_image(image: Image.Image, format="JPEG") -> str:
        buffered = io.BytesIO()
        # Handle both PIL Image and numpy array inputs
        if isinstance(image, (list, np.ndarray)):
            # Convert numpy array to PIL Image
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
        """Add padding around mask using dilation"""
        # Convert PIL to numpy array
        mask_np = np.array(mask_image.convert('L'))
        
        # Create kernel for dilation
        kernel = np.ones((padding, padding), np.uint8)
        
        # Dilate mask
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        
        # Convert back to PIL
        return Image.fromarray(dilated_mask)

class CaptionUtils:
    @staticmethod
    def parse_json_response(response_text: str) -> dict:
        try:
            if response_text.startswith("```json\n"):
                json_string = response_text.split("```json\n")[1].split("```")[0]
            else:
                json_string = response_text.replace('\n', '')
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse model response: {str(e)}"
            )

    @staticmethod
    def get_conversation_template(image_base64: str) -> list:
        return [
            {
                "role": "system",
                "content": """You are a furniture expert. Analyze images and provide descriptions in this exact JSON structure:
                                {
                                    "type": "<must be one of: bed, chair, table, sofa>",
                                    "style": "<describe overall style>",
                                    "color": "<describe main color>",
                                    "material": "<describe primary material>",
                                    "shape": "<describe general shape>",
                                    "details": "<describe one decorative feature>",
                                    "room_type": "<specify room type>",
                                    "price_range": "<specify price range>"
                                }
                            Focus on maintaining this exact structure while providing relevant descriptions."""
            },
            {
                "role": "assistant",
                "content": "I will analyze the image and respond with a valid JSON object following the exact schema."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": "Describe this furniture piece in JSON format."
                    }
                ]
            }
        ]