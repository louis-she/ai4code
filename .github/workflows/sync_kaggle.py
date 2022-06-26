from genericpath import exists
import os
import tarfile
import os.path


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


os.makedirs("./kaggle_sync/", exist_ok=True)
make_tarfile("./kaggle_sync/code.tar.gz", f"{os.getenv('GITHUB_WORKSPACE')}/ai4code")
metadata = """{
  "title": "ai4code",
  "id": "snaker/ai4code",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
"""

open("./kaggle_sync/dataset-metadata.json", "w").write(metadata)
os.system("cd ./kaggle_sync && kaggle datasets create")
