import pytest
import base64
import io
import numpy as np
from PIL import Image
from fastapi import HTTPException

from app.utils.image_utils import ImageUtils


class TestImageUtils:
    
    def test_encode_image_pil(self):
        """Test encoding PIL Image to base64"""
        img = Image.new('RGB', (100, 100), color='red')
        encoded = ImageUtils.encode_image(img)
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Verify it's valid base64
        decoded_bytes = base64.b64decode(encoded)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (100, 100)
    
    def test_encode_image_numpy_array(self):
        """Test encoding numpy array to base64"""
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :] = [255, 0, 0]  # Red image
        
        encoded = ImageUtils.encode_image(img_array)
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_encode_image_with_format(self):
        """Test encoding with different formats"""
        img = Image.new('RGB', (100, 100), color='blue')
        
        # Test JPEG format
        encoded_jpg = ImageUtils.encode_image(img, format="JPEG")
        assert isinstance(encoded_jpg, str)
        
        # Test PNG format
        encoded_png = ImageUtils.encode_image(img, format="PNG")
        assert isinstance(encoded_png, str)
    
    def test_decode_image_valid(self):
        """Test decoding valid base64 image"""
        # Create a test image and encode it
        original_img = Image.new('RGB', (50, 50), color='green')
        buffer = io.BytesIO()
        original_img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        
        # Decode it back
        decoded_img = ImageUtils.decode_image(encoded)
        
        assert isinstance(decoded_img, Image.Image)
        assert decoded_img.size == (50, 50)
    
    def test_decode_image_invalid_base64(self):
        """Test decoding invalid base64 string"""
        invalid_b64 = "this_is_not_valid_base64"
        
        with pytest.raises(HTTPException) as exc_info:
            ImageUtils.decode_image(invalid_b64)
        
        assert exc_info.value.status_code == 400
        assert "Invalid image data" in str(exc_info.value.detail)
    
    def test_decode_image_invalid_image_data(self):
        """Test decoding valid base64 but invalid image data"""
        invalid_image_b64 = base64.b64encode(b"this is not image data").decode('utf-8')
        
        with pytest.raises(HTTPException) as exc_info:
            ImageUtils.decode_image(invalid_image_b64)
        
        assert exc_info.value.status_code == 400
        assert "Invalid image data" in str(exc_info.value.detail)
    
    def test_add_mask_padding(self):
        """Test adding padding to mask image"""
        # Create a small white square in the center of a black image
        mask = Image.new('L', (100, 100), color=0)  # Black background
        mask.paste(255, (40, 40, 60, 60))  # White square in center
        
        padded_mask = ImageUtils.add_mask_padding(mask, padding=10)
        
        assert isinstance(padded_mask, Image.Image)
        assert padded_mask.size == mask.size
        
        # Convert to numpy to check if dilation occurred
        original_array = np.array(mask)
        padded_array = np.array(padded_mask)
        
        # The padded mask should have more white pixels
        assert np.sum(padded_array > 0) > np.sum(original_array > 0)
    
    def test_add_mask_padding_zero_padding(self):
        """Test adding zero padding (should return similar image)"""
        mask = Image.new('L', (50, 50), color=128)
        
        padded_mask = ImageUtils.add_mask_padding(mask, padding=0)
        
        # With zero padding, images should be very similar
        original_array = np.array(mask)
        padded_array = np.array(padded_mask)
        
        # Allow for small differences due to dilation with 0-size kernel
        assert np.allclose(original_array, padded_array, atol=10)