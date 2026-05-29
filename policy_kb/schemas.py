from pydantic import BaseModel, Field
import uuid

class PolicyChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this specific text chunk")
    document_id: str = Field(..., description="Name or unique ID of the source PDF (e.g., 'FDA_Guidance_2023')")
    page_number: int = Field(..., description="The page number where this text was found")
    section_hierarchy: str | None = Field(default = None, description="The clause hierarchy")
    is_table: bool = Field(default=False, description= "Whether the chunk is a table")
    bounding_box: list[list[float]] | None = Field(default=None, description= "Coordinates of the table (if chunk is a table)")
    text: str = Field(..., description="The actual text content of the chunk")