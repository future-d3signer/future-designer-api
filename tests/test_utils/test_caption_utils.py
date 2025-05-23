import pytest
import json
from fastapi import HTTPException

from app.utils.caption_utils import CaptionUtils


class TestCaptionUtils:
    
    def test_parse_json_response_clean_json(self):
        """Test parsing clean JSON response"""
        clean_json = '{"type": "chair", "style": "modern", "color": "red"}'
        
        result = CaptionUtils.parse_json_response(clean_json)
        
        assert isinstance(result, dict)
        assert result["type"] == "chair"
        assert result["style"] == "modern"
        assert result["color"] == "red"
    
    def test_parse_json_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks"""
        markdown_json = '''```json
{
    "type": "table",
    "style": "vintage",
    "color": "brown",
    "material": "wood"
}
```'''
        
        result = CaptionUtils.parse_json_response(markdown_json)
        
        assert isinstance(result, dict)
        assert result["type"] == "table"
        assert result["style"] == "vintage"
        assert result["color"] == "brown"
        assert result["material"] == "wood"
    
    def test_parse_json_response_with_newlines(self):
        """Test parsing JSON with newlines"""
        json_with_newlines = '''{\n"type": "sofa",\n"style": "contemporary"\n}'''
        
        result = CaptionUtils.parse_json_response(json_with_newlines)
        
        assert isinstance(result, dict)
        assert result["type"] == "sofa"
        assert result["style"] == "contemporary"
    
    def test_parse_json_response_invalid_json(self):
        """Test parsing invalid JSON raises HTTPException"""
        invalid_json = '{"type": "chair", "style": invalid}'
        
        with pytest.raises(HTTPException) as exc_info:
            CaptionUtils.parse_json_response(invalid_json)
        
        assert exc_info.value.status_code == 500
        assert "Failed to parse model response" in str(exc_info.value.detail)
    
    def test_parse_json_response_empty_string(self):
        """Test parsing empty string raises HTTPException"""
        with pytest.raises(HTTPException) as exc_info:
            CaptionUtils.parse_json_response("")
        
        assert exc_info.value.status_code == 500
        assert "Failed to parse model response" in str(exc_info.value.detail)
    
    def test_get_conversation_template(self):
        """Test getting conversation template"""
        test_image_b64 = "test_base64_string"
        
        conversation = CaptionUtils.get_conversation_template(test_image_b64)
        
        assert isinstance(conversation, list)
        assert len(conversation) == 3
        
        # Check system message
        assert conversation[0]["role"] == "system"
        assert "furniture expert" in conversation[0]["content"].lower()
        
        # Check assistant message
        assert conversation[1]["role"] == "assistant"
        
        # Check user message
        assert conversation[2]["role"] == "user"
        assert len(conversation[2]["content"]) == 2
        
        # Check image URL in user message
        image_content = conversation[2]["content"][0]
        assert image_content["type"] == "image_url"
        assert f"data:image/jpeg;base64,{test_image_b64}" in image_content["image_url"]["url"]
        
        # Check text content in user message
        text_content = conversation[2]["content"][1]
        assert text_content["type"] == "text"
        assert "json format" in text_content["text"].lower()
    
    def test_conversation_template_structure(self):
        """Test that conversation template has correct JSON schema requirements"""
        conversation = CaptionUtils.get_conversation_template("dummy_b64")
        
        system_content = conversation[0]["content"]
        
        # Check that required fields are mentioned in system prompt
        required_fields = ["type", "style", "color", "material", "shape", "details", "room_type", "price_range"]
        for field in required_fields:
            assert field in system_content
        
        # Check that valid furniture types are mentioned
        valid_types = ["bed", "chair", "table", "sofa"]
        for furniture_type in valid_types:
            assert furniture_type in system_content