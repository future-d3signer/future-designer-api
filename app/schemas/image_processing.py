from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict


class FurnitureType(str, Enum):
    BED = "bed"
    CHAIR = "chair"
    TABLE = "table"
    SOFA = "sofa"


class FurnitureDescription(BaseModel):
    type: FurnitureType
    style: str
    color: str
    material: str
    details: str
    room_type: str


class CaptionRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded image data")


class StyleRequest(BaseModel):
    style_image: str = Field(..., description="Base64 encoded depth image")
    style: str = Field(..., description="Style identifier for the transfer")


class ReplaceRequest(BaseModel):
    style: str = Field(..., description="Style identifier for the transfer")
    mask_image: str = Field(..., description="Base64 encoded mask image")
    adapter_image: str = Field(..., description="Base64 encoded furniture image")


class DepthRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded image data")


class StyleResponse(BaseModel):
    generated_image: str = Field(..., description="Base64 encoded generated image")


class DepthResponse(BaseModel):
    depth_image: str


class FurnitureItem(BaseModel):
    caption: FurnitureDescription
    mask: str = Field(..., description="Base64 encoded mask image")
    box: str = Field(..., description="Base64 encoded mask image")
    furniture_image: str = Field(..., description="Base64 encoded furniture image")


class CaptionResponse(BaseModel):
    furniture: Dict[str, FurnitureItem]


class TransparencyRequest(BaseModel):
    furniture_image: str = Field(..., description="Base64 encoded furniture image with white background")


class TransparencyResponse(BaseModel):
    transparent_image: str = Field(..., description="Base64 encoded furniture image with transparent background")


class CompositeRequest(BaseModel):
    room_image: str  
    furniture_image: str  
    position: Dict[str, int]  
    size: Dict[str, int]  
