from dataclasses import dataclass, field
from typing import List


@dataclass
class SampleRawData:
    rawdata_name : str
    rawdata_path : str = ''
    frame_index_list: List[int] = field(default_factory=list)
    pcd_files: List[str] = field(default_factory=list)

    def __eq__(self, compared_object: object) -> bool:
        return self.rawdata_name == compared_object.rawdata_name
