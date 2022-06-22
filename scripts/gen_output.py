import os
import sys
import typing
from termcolor import colored
from fire import Fire
import torch
from ai4code import utils


def main(
    checkpoint: str,
    output: str = None,
    use_cwd: bool = False,
):
    global ai4code
    state = torch.load(checkpoint, map_location='cpu')
    params = state["params"]

    datasets_arguments = ["anchor_size", "max_len", "split_len", "distil_context", ""]


Fire(main)
