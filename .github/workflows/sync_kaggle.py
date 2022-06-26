from genericpath import exists
import os
import tarfile
import os.path
from pathlib import Path


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


os.makedirs("./kaggle_sync/", exist_ok=True)
make_tarfile("./kaggle_sync/code.tar.gz", f"{os.getenv('GITHUB_WORKSPACE')}/ai4code")
metadata = """{
  "title": "chenglu_ai4code_source",
  "id": "snaker/ai4code",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
"""

Path("./kaggle_sync/dataset-metadata.json").write_text(metadata)
os.system("ls -lht ./kaggle_sync")
os.system("cat ./kaggle_sync/dataset-metadata.json")
os.system("cd ./kaggle_sync && kaggle datasets version -m 'new'")
