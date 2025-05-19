from pydantic import BaseModel


class SearchRequest(BaseModel):
    type: str = ""
    style: str = ""
    color: str = ""
    material: str = ""
    details: str = ""
    room_type: str = ""


class URLRequest(BaseModel):
    url: str