import pytest
import requests
from unittest.mock import Mock, patch
from requests.exceptions import RequestException

from app.services.web_service import WebService


class TestWebService:
    
    @pytest.fixture
    def web_service(self):
        return WebService()
    
    @patch('app.services.web_service.requests.get')
    def test_proxy_image_success(self, mock_get, web_service):
        """Test successful image proxying"""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = b'fake_image_data'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = web_service.proxy_image("https://example.com/image.jpg")
        
        # Check base64 encoding
        import base64
        expected = base64.b64encode(b'fake_image_data').decode('utf-8')
        assert result == expected
        
        # Verify request was made with proper headers
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://example.com/image.jpg"
        assert "User-Agent" in call_args[1]["headers"]
        assert "Referer" in call_args[1]["headers"]
    
    @patch('app.services.web_service.requests.get')
    def test_proxy_image_http_error(self, mock_get, web_service):
        """Test proxy image with HTTP error"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = RequestException("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(RequestException):
            web_service.proxy_image("https://example.com/nonexistent.jpg")
    
    @patch('app.services.web_service.requests.get')
    @patch('app.services.web_service.BeautifulSoup')
    def test_scrape_image_links_no_gallery(self, mock_soup, mock_get, web_service):
        """Test scraping when no gallery is found"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = b'<html>no gallery here</html>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup parsing - no gallery found
        mock_soup_instance = Mock()
        mock_soup.return_value = mock_soup_instance
        mock_soup_instance.find.return_value = None  # No gallery div found
        
        with pytest.raises(ValueError, match="No image gallery thumbnails found"):
            web_service.scrape_image_links("https://example.com/no-gallery")
    
    @patch('app.services.web_service.requests.get')
    def test_scrape_image_links_http_error(self, mock_get, web_service):
        """Test scraping with HTTP error"""
        mock_get.side_effect = RequestException("Connection error")
        
        with pytest.raises(RequestException):
            web_service.scrape_image_links("https://example.com/error")
    
    @patch('app.services.web_service.requests.get')
    @patch('app.services.web_service.BeautifulSoup')
    def test_scrape_image_links_empty_gallery(self, mock_soup, mock_get, web_service):
        """Test scraping when gallery exists but has no images"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = b'<html>empty gallery</html>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup parsing
        mock_soup_instance = Mock()
        mock_soup.return_value = mock_soup_instance
        
        # Mock gallery div exists but no images
        mock_gallery_div = Mock()
        mock_soup_instance.find.return_value = mock_gallery_div
        mock_gallery_div.find_all.return_value = []  # No img tags
        
        result = web_service.scrape_image_links("https://example.com/empty-gallery")
        
        assert result == []