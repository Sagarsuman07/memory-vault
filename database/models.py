from dataclasses import dataclass


@dataclass
class Memory:

    id: str
    user_id: str
    memory_type: str
    title: str
    file_name: str
    file_path: str
    created_at: str
    updated_at: str