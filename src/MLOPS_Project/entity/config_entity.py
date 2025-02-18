from dataclasses import dataclass
from pathlib import Path


## what type of input should a class represent
@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
