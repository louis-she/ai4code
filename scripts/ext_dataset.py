import random
from tqdm import tqdm
import gzip
import json
from pathlib import Path
import fire
import secrets

import pandas


def main(source: str, dest: str, gziped: bool = False):
    source: Path = Path(source)
    dest:Path = Path(dest)

    train = dest / "train"

    orders_csv_path = dest / "train_orders.csv"
    if orders_csv_path.exists():
        orders_csv = pandas.read_csv(orders_csv_path.as_posix())
    else:
        orders_csv = pandas.DataFrame()

    cell_orders = {"id": [], "cell_order": []}
    for item in tqdm(source.glob("*"), total=len(list(source.glob("*")))):
        try:
            sample_id = secrets.token_hex(nbytes=8)
            with item.open("rb") as f:
                bytes = f.read()
                if gziped:
                    bytes = gzip.decompress(bytes)
                data = json.loads(bytes)
            sample = {
                "cell_type": {},
                "source": {}
            }
            cell_order = []
            markdown_temp = []
            cells = [cell for cell in data["cells"] if cell["cell_type"] in ["markdown", "code"]]
            if len(cells) == 0:
                continue
            for cell in cells:
                cell_type = cell["cell_type"]
                if cell_type not in ['markdown', 'code']:
                    continue
                cell_id = secrets.token_hex(nbytes=4)
                cell_order.append(cell_id)
                if isinstance(cell["source"], str):
                    cell_source = cell["source"]
                else:
                    cell_source = "\n".join(cell["source"])
                if cell_type == "code":
                    sample['cell_type'][cell_id] = cell_type
                    sample['source'][cell_id] = cell_source
                elif cell_type == "markdown":
                    markdown_temp.append((cell_id, cell_type, cell_source))
            random.shuffle(markdown_temp)
            for cell_id, cell_type, cell_source in markdown_temp:
                sample['cell_type'][cell_id] = cell_type
                sample['source'][cell_id] = cell_source
            with (train / (sample_id + ".json")).open("w") as f:
                json.dump(sample, f)
            cell_orders['id'].append(sample_id)
            cell_orders['cell_order'].append(" ".join(cell_order))
        except Exception as e:
            print(f"error: {item} - {e}")
    orders_csv = pandas.concat([orders_csv, pandas.DataFrame.from_dict(cell_orders)])
    orders_csv.to_csv(dest / "train_orders.csv", index=False)


fire.Fire(main)