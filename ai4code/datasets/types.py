from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Sample:
    id: str
    sources: Dict[str, str]
    ancestor: str
    parent: str
    orders: List[str]
    content_len: int
    cell_keys: Dict[str, str]
    cell_lens: Dict[str, int]
    cell_ranks: Dict[str, float]
    cell_ranks_normed: Dict[str, float]
    cell_types: Dict[str, str]
    cell_encodes: Dict[str, List[int]]

    # Statistical
    markdown_cell_count: int
    code_cell_count: int
    code_ratio: int

    percentile_cell_lens: List[float]
    mean_cell_lens: float

    # data split
    fold: Optional[int] = None

    @property
    def cell_type_order(self):
        return [self.cell_types[key] for key in self.orders]


@dataclass
class SpecialTokenID:
    hash_id: int
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int
    unk_token_id: int
