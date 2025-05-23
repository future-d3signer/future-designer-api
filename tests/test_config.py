import pytest
import os
from unittest.mock import patch

from app.core.config import Settings, settings


class TestConfig:
    
    def test_settings_defaults(self):
        """Test that settings have expected default values"""
        test_settings = Settings()
        
        # Test paths have defaults
        assert test_settings.DINO_CONFIG_PATH.endswith("GroundingDINO_SwinT_OGC.py")
        assert test_settings.DINO_WEIGHTS_PATH.endswith("groundingdino_swint_ogc.pth")
        assert test_settings.PROMPTS_FILE_PATH == "styles.json"
        assert test_settings.SAM2_CHECKPOINT_PATH.endswith("sam2.1_hiera_large.pt")
        assert test_settings.SAM2_MODEL_CONFIG == "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    @patch.dict(os.environ, {
        'MILVUS_URL': 'test_url',
        'MILVUS_TOKEN': 'test_token'
    })
    def test_settings_from_env(self):
        """Test that settings load from environment variables"""
        test_settings = Settings()
        
        assert test_settings.MILVUS_URL == 'test_url'
        assert test_settings.MILVUS_TOKEN == 'test_token'
    
    @patch.dict(os.environ, {
        'DINO_CONFIG_PATH': '/custom/path/config.py',
        'SAM2_CHECKPOINT_PATH': '/custom/sam/checkpoint.pt'
    })
    def test_settings_custom_paths(self):
        """Test that custom paths can be set via environment"""
        test_settings = Settings()
        
        assert test_settings.DINO_CONFIG_PATH == '/custom/path/config.py'
        assert test_settings.SAM2_CHECKPOINT_PATH == '/custom/sam/checkpoint.pt'
    
    def test_settings_instance_exists(self):
        """Test that global settings instance exists"""
        assert settings is not None
        assert isinstance(settings, Settings)
    
    def test_settings_config_class(self):
        """Test settings configuration"""
        test_settings = Settings()
        
        # Test that config is properly set
        assert hasattr(test_settings.Config, 'env_file')
        assert test_settings.Config.env_file == '.env'
        assert test_settings.Config.env_file_encoding == 'utf-8'
        assert test_settings.Config.extra == 'ignore'