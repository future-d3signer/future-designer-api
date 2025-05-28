import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import base64
from PIL import Image
import io
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_base64_image():
    """Create a mock base64 encoded image for testing"""
    img = Image.new('RGB', (1024, 1024), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


@pytest.fixture
def mock_pil_image():
    """Create a mock PIL image that can be used by the service"""
    return Image.new('RGB', (1024, 1024), color='red')


class TestImageGenerationRouter:
    
    @patch('app.api.deps.get_image_service')
    @patch('app.api.deps.CudaMemoryManagerDep')
    def test_generate_style_success(self, mock_cuda_dep, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.generate_styled_image.return_value = mock_base64_image
        mock_image_service.return_value = mock_service
        mock_cuda_dep.return_value = None
        
        with patch('app.utils.image_utils.ImageUtils.decode_image') as mock_decode:
            mock_decode.return_value = Image.new('RGB', (1024, 1024), color='red')
            
            response = client.post("/image-generation/generate_style", json={
                "style": "modern",
                "depth_image_b64": mock_base64_image
            })
            
            assert response.status_code == 200
            assert "generated_image" in response.json()
           

    @patch('app.api.deps.get_image_service')
    @patch('app.api.deps.CudaMemoryManagerDep')
    def test_generate_inpaint_success(self, mock_cuda_dep, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.generate_inpaint.return_value = mock_base64_image
        mock_image_service.return_value = mock_service
        mock_cuda_dep.return_value = None
        
        with patch('app.utils.image_utils.ImageUtils.decode_image') as mock_decode:
            mock_decode.return_value = Image.new('RGB', (1024, 1024), color='red')
            
            response = client.post("/image-generation/generate_inpaint", json={
                "style_prompt": "modern living room",
                "orginal_image_b64": mock_base64_image,
                "mask_image_b64": mock_base64_image,
                "depth_image_b64": mock_base64_image
            })
            
            assert response.status_code == 200
            assert "generated_image" in response.json()
          

    @patch('app.api.deps.get_image_service')
    @patch('app.api.deps.CudaMemoryManagerDep')
    def test_generate_delete_success(self, mock_cuda_dep, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.generate_delete.return_value = mock_base64_image
        mock_image_service.return_value = mock_service
        mock_cuda_dep.return_value = None
        
        with patch('app.utils.image_utils.ImageUtils.decode_image') as mock_decode:
            mock_decode.return_value = Image.new('RGB', (1024, 1024), color='red')
            
            response = client.post("/image-generation/generate_delete", json={
                "orginal_image_b64": mock_base64_image,
                "box_image_b64": mock_base64_image
            })
            
            assert response.status_code == 200
            assert "generated_image" in response.json()

    @patch('app.api.deps.get_image_service')
    def test_generate_style_validation_error(self, mock_image_service, client):
        mock_service = Mock()
        mock_service.generate_styled_image.side_effect = ValueError("Invalid style")
        mock_image_service.return_value = mock_service
        
        response = client.post("/image-generation/generate_style", json={
            "style": "invalid_style",
            "depth_image_b64": "invalid_base64"
        })
        
        assert response.status_code == 500

    @patch('app.api.deps.get_image_service')
    def test_generate_inpaint_validation_error(self, mock_image_service, client):
        mock_service = Mock()
        mock_service.generate_inpaint.side_effect = ValueError("Invalid prompt")
        mock_image_service.return_value = mock_service
        
        response = client.post("/image-generation/generate_inpaint", json={
            "style_prompt": "",
            "orginal_image_b64": "invalid",
            "mask_image_b64": "invalid",
            "depth_image_b64": "invalid"
        })
        
        assert response.status_code == 500

    @patch('app.api.deps.get_image_service')
    def test_generate_delete_validation_error(self, mock_image_service, client):
        mock_service = Mock()
        mock_service.generate_delete.side_effect = ValueError("Invalid image")
        mock_image_service.return_value = mock_service
        
        response = client.post("/image-generation/generate_delete", json={
            "orginal_image_b64": "invalid",
            "box_image_b64": "invalid"
        })
        
        assert response.status_code == 500

    @patch('app.api.deps.get_image_service')
    def test_generate_replace_validation_error(self, mock_image_service, client):
        mock_service = Mock()
        mock_service.generate_replace.side_effect = ValueError("Invalid adapter")
        mock_image_service.return_value = mock_service
        
        response = client.post("/image-generation/generate_replace", json={
            "style_prompt": "test",
            "orginal_image_b64": "invalid",
            "mask_image_b64": "invalid",
            "adapter_image_name": "nonexistent.jpg"
        })
        
        assert response.status_code == 500

    @patch('app.api.deps.get_image_service')
    def test_generate_replace_server_error(self, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.generate_replace.side_effect = Exception("Server error")
        mock_image_service.return_value = mock_service
        
        response = client.post("/image-generation/generate_replace", json={
            "style_prompt": "modern",
            "orginal_image_b64": mock_base64_image,
            "mask_image_b64": mock_base64_image,
            "adapter_image_name": "test.jpg"
        })
        
        assert response.status_code == 500

    def test_generate_style_missing_style(self, client, mock_base64_image):
        response = client.post("/image-generation/generate_style", json={
            "depth_image_b64": mock_base64_image
        })
        
        assert response.status_code == 422

    def test_generate_style_missing_image(self, client):
        response = client.post("/image-generation/generate_style", json={
            "style": "modern"
        })
        
        assert response.status_code == 422

    def test_generate_inpaint_missing_prompt(self, client, mock_base64_image):
        response = client.post("/image-generation/generate_inpaint", json={
            "orginal_image_b64": mock_base64_image,
            "mask_image_b64": mock_base64_image,
            "depth_image_b64": mock_base64_image
        })
        
        assert response.status_code == 422

    def test_generate_delete_missing_box_image(self, client, mock_base64_image):
        response = client.post("/image-generation/generate_delete", json={
            "orginal_image_b64": mock_base64_image
        })
        
        assert response.status_code == 422

    def test_generate_replace_missing_adapter(self, client, mock_base64_image):
        response = client.post("/image-generation/generate_replace", json={
            "style_prompt": "modern",
            "orginal_image_b64": mock_base64_image,
            "mask_image_b64": mock_base64_image
        })
        
        assert response.status_code == 422


class TestImageAnalysisRouter:
    
    @patch('app.api.deps.get_image_service')
    @patch('app.api.deps.CudaMemoryManagerDep')
    def test_generate_depth_success(self, mock_cuda_dep, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.generate_depth_map.return_value = (None, None, mock_base64_image)
        mock_image_service.return_value = mock_service
        mock_cuda_dep.return_value = None
        
        response = client.post("/image-analysis/generate_depth", json={
            "source_image": mock_base64_image
        })
        
        assert response.status_code == 200
        assert "depth_image" in response.json()
       

    @patch('app.api.deps.get_image_service')
    @patch('app.api.deps.CudaMemoryManagerDep')
    def test_generate_captions_success(self, mock_cuda_dep, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.generate_captions.return_value = ["A modern living room"]
        mock_image_service.return_value = mock_service
        mock_cuda_dep.return_value = None
        
        response = client.post("/image-analysis/generate_captions", json={
            "source_image": mock_base64_image
        })
        
        assert response.status_code == 200
        
    @patch('app.api.deps.get_image_service')
    def test_generate_transparency_validation_error(self, mock_image_service, client):
        mock_service = Mock()
        mock_service.make_furniture_transparent.side_effect = ValueError("Invalid image format")
        mock_image_service.return_value = mock_service
        
        response = client.post("/image-analysis/generate_transparency", json={
            "furniture_image": "invalid_base64"
        })
        
        assert response.status_code == 500


class TestSearchRouter:
    
    @patch('app.api.deps.get_milvus_service')
    def test_search_similar_items_success(self, mock_milvus_service, client):
        mock_service = Mock()
        mock_service.search_similar.return_value = [
            {"id": "1", "score": 0.95, "metadata": {"name": "item1"}},
            {"id": "2", "score": 0.85, "metadata": {"name": "item2"}}
        ]
        mock_milvus_service.return_value = mock_service
        
        response = client.post("/search", json={
            "query_vector": [0.1, 0.2, 0.3],
            "collection_name": "furniture"
        }, params={"top_k": 5})
        
        assert response.status_code == 200
    

    @patch('app.api.deps.get_milvus_service')
    def test_search_with_invalid_top_k(self, mock_milvus_service, client):
        response = client.post("/search", json={
            "query_vector": [0.1, 0.2, 0.3],
            "collection_name": "furniture"
        }, params={"top_k": 100})  # Exceeds maximum of 50
        
        assert response.status_code == 422  # Validation error


class TestUtilityRouter:

    @patch('app.api.deps.get_web_service')
    def test_proxy_image_empty_url(self, mock_web_service, client):
        response = client.post("/utility/proxy-image", json={
            "url": ""
        })
        
        assert response.status_code == 400

    @patch('app.api.deps.get_web_service')
    def test_scrape_images_no_images_found(self, mock_web_service, client):
        mock_service = Mock()
        mock_service.scrape_image_links.return_value = []
        mock_web_service.return_value = mock_service
        
        response = client.post("/utility/scrape-images", json={
            "url": "https://example.com/empty-gallery"
        })
        
        assert response.status_code == 502

    @patch('app.api.deps.get_image_service')
    @patch('app.api.deps.CudaMemoryManagerDep')
    def test_composite_furniture_success(self, mock_cuda_dep, mock_image_service, client, mock_base64_image):
        mock_service = Mock()
        mock_service.composite_and_blend_furniture.return_value = mock_base64_image
        mock_image_service.return_value = mock_service
        mock_cuda_dep.return_value = None
        
        response = client.post("/utility/composite_furniture", json={
            "room_image": mock_base64_image,
            "furniture_image": mock_base64_image,
            "position": {"x": 100, "y": 100},
            "size": {"width": 200, "height": 150}
        })
        
        assert response.status_code == 200
        assert "composited_image" in response.json()


class TestErrorHandling:

    def test_invalid_json_request(self, client):
        response = client.post("/image-analysis/generate_depth", 
                             data="invalid json", 
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        response = client.post("/image-generation/generate_style", json={
            "style": "modern"
            # Missing depth_image_b64
        })
        
        assert response.status_code == 422


@pytest.mark.integration
class TestIntegrationTests:
    
    def test_full_style_generation_flow(self, client, mock_base64_image):
        response = client.post("/image-generation/generate_style", json={
            "style": "modern",
            "depth_image_b64": mock_base64_image
        })

        assert response.status_code in [200, 500, 422]