import pwd
import re
import subprocess
import os
from termcolor import colored


def cprint(text, color):
    print(colored(text, color))

try:
    res = subprocess.run("git diff-index --quiet HEAD", shell=True, check=True, capture_output=True)
except subprocess.CalledProcessError as e:
    cprint(f"working directory may not be clean, please submit all the changes to do training: {e}", "red")
    exit(1)

try:
    res = subprocess.run("git rev-parse HEAD", shell=True, check=True, capture_output=True)
except subprocess.CalledProcessError as e:
    cprint(f"get git commit failed with {e}", "red")
    exit(1)

git_commit = res.stdout.decode().replace("\n", "")
train_dir = f"/home/featurize/work/ai4code/_runs/{git_commit}"

if os.path.exists(train_dir):
    cprint("Training folder exists, will skip git clone", "yellow")
else:
    try:
        res = subprocess.run(f"git clone /home/featurize/work/ai4code {train_dir}", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        cprint(f"git clone errored: {e}", "red")
        exit(1)


def train(command):
    command = re.sub("python", f"python --git_commits {git_commit} ", command)
    cprint(command, "green")
    # subprocess.run(command, shell=True, pwd=train_dir)


if os.environ["machine_name"] == "main":
    train("""python
    --pretrained_path /home/featurize/distilbert-base-uncased/distilbert-base-uncased \
    --code tune_params \
    --override \
    --lr 0.00005 \
    --batch_size 32 \
    --max_epochs 3 \
    --optimizer Adam \
    --context_cells_token_size 18 \
    --context_stride 3 \
    --cell_token_size 64 \
    --cell_stride 1 \
    --dataset_suffix v4 \
""")
