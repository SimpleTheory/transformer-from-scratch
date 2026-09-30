from .base_objects import *
from .local_dataset_code import *
from .trainer import *
from pathlib import Path

def project_root(relative_path: str | Path | None = None) -> Path:
    current_file_path = Path(__file__).resolve()
    # transformer_from_scratch -> src -> base
    result = current_file_path.parent.parent.parent
    if relative_path:
        return result / Path(relative_path)
    return result
