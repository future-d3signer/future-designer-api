import json

from fastapi import HTTPException


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