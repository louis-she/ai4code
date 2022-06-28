import torch
from typing import Dict

from transformers import LongformerTokenizer

from ai4code.datasets import Sample


class LongFormerDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, Sample]):
        self.data = data
        self.tokenizer = LongformerTokenizer.from_pretrained(
            "allenai/longformer-base-4096"
        )
        self.all_cells = []
        for sample in list(self.data.values()):
            for k, cell_id in enumerate(sample.cell_keys):
                self.all_cells.append((sample.id, cell_id, k))

    def __len__(self):
        return len(self.all_cells)

    def __getitem__(self, index: int):
        sample_id, cell_id, cell_index = self.all_cells[index]
        sample = self.data[sample_id]
        content = sample.sources[cell_index] + "\n\n\n\n" + "\n\n".join(sample.sources)
        label = torch.tensor(sample.orders.index(cell_id) / len(sample.orders))
        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = torch.tensor(inputs["input_ids"]).long()
        mask = torch.tensor(inputs["attention_mask"]).long()
        return ids, mask, label, sample_id, cell_id