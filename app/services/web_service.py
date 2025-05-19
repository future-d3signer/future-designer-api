import requests
from bs4 import BeautifulSoup
import base64

class WebService:
    def proxy_image(self, url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.otodom.pl/" # Be mindful of ToS
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Will raise for 4xx/5xx
        return base64.b64encode(response.content).decode('utf-8')

    def scrape_image_links(self, url: str) -> list[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        gallery_div = soup.find('div', {'data-sentry-element': 'GalleryMainContainer'})
        if gallery_div:
            image_tags = gallery_div.find_all('img')
            return [img['src'] for img in image_tags if 'src' in img.attrs]
        raise ValueError("No image gallery thumbnails found.")