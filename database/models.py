from dataclasses import dataclass
from typing import Optional


@dataclass
class Memory:
    id: str
    user_id: str
    memory_type: str
    title: str
    file_name: str
    file_path: str
    extracted_text: Optional[str]
    created_at: str
    updated_at: str