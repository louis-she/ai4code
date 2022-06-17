import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from termcolor import colored

import fire
import torch
import torch.nn.functional as F
from aim.pytorch_ignite import AimLogger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import RunningAverage
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer

from ai4code import datasets, metrics, models
from ai4code.utils import SerializableDict
import ignite.distributed as idist
from transformers import logging

logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG_DIR = Path("/home/featurize/ai4code")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(
    code: str,
    pretrained_path: str,
    dataset_suffix: str,
    git_commit: str,
    lr: int = 3e-5,
    max_epochs: int = 5,
    batch_size: int = 256,
    seed: int = 777,
    resume: str = None,
    load_model: str = None,
    override: bool = False,
    optimizer: str = "AdamW",
    testing: bool = False,
    saving_checkpoint: bool = True,
    num_workers: int = 2,
    train_folds: Tuple[int] = (1,),
    val_folds: Tuple[int] = (0,),
    evaluate_every: int = 1,
    with_scheduler: bool = True,
    ordered_context_ratio: float = 0.0,
    validate_with_ordered: bool = False,
    split_len: int = 8,
    accumulation_steps: int = 1,
    with_lm: bool = False,
    pair_lm: bool = False,
    # dataset temp
    negative_ratio: float = 0.5,
    cell_token_size: int = 64,
    cell_stride: int = 1,
    context_stride: int = 1,
    max_len: int = 256,
    train_num_samples: int = None,
    val_num_samples: int = None,
    dropout: float = 0.2,
    train_all_cells: bool = False,
    distil_context: str = None,
    tokenizer_pretrained_path: str = None,
):
    params = SerializableDict(locals())
    torch.manual_seed(seed)
    device = idist.device()
    rank = idist.get_local_rank()

    if tokenizer_pretrained_path is None:
        tokenizer_pretrained_path = pretrained_path

    max_epochs = max_epochs * len(train_folds)

    if testing:
        train_num_samples = 100
        val_num_samples = 100
        code = "test_" + code
        override = True

    code_dir = LOG_DIR / code
    if code_dir.exists() and not override:
        exit(
            f"Code dir {code_dir} exists! use --override to force override it or change another code name"
        )

    val_data = pickle.load(open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/0.pkl", "rb"))
    if val_num_samples is not None:
        val_data = {k: v for k, v in list(val_data.items())[:val_num_samples]}

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_pretrained_path, do_lower_case=True, use_fast=True
    )
    vocab_len = len(tokenizer)

    special_tokens = datasets.SpecialTokenID(
        hash_id=tokenizer.encode("#", add_special_tokens=False)[0],
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )

    del tokenizer

    current_train_fold_idx = 0
    current_lm_fold_idx = 0

    def reset_fold_idx():
        nonlocal current_train_fold_idx, current_lm_fold_idx
        current_train_fold_idx = 0
        current_lm_fold_idx = 0

    def create_dataset(data):
        return datasets.RankDatasetWithSplits(
            data,
            special_tokens=special_tokens,
            cell_token_size=cell_token_size,
            cell_stride=cell_stride,
            context_stride=context_stride,
            max_len=max_len,
            split_len=split_len,
            distil_context=distil_context,
        )

    def get_next_loader():
        # 由于内存原因，每个 DataLoader 只能遍历单个 Fold
        # 因此一轮表示单个 Fold 的训练，每次训练完毕后手动
        # 把 Loader 改为下一个 Fold
        nonlocal current_train_fold_idx
        fold = train_folds[current_train_fold_idx % len(train_folds)]
        if rank == 0:
            print(colored(f"generate loader of fold {fold}", "green"))
        train_data = pickle.load(open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/{fold}.pkl", "rb"))
        if train_num_samples is not None:
            train_data = {k: v for k, v in list(train_data.items())[:train_num_samples]}

        current_train_fold_idx += 1

        return idist.auto_dataloader(
            create_dataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )

    def get_next_lm_loader():
        nonlocal current_lm_fold_idx
        fold = train_folds[current_train_fold_idx % len(train_folds)]
        lm_train_data = pickle.load(open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/{fold}.pkl", "rb"))
        current_lm_fold_idx += 1

        return iter(idist.auto_dataloader(
            datasets.PairLMDataset(lm_train_data, special_tokens, max_len),
            batch_size=batch_size,
            shuffle=True,
        ))

    val_loader = idist.auto_dataloader(
        create_dataset(val_data),
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if pair_lm:
        pair_lm_loader = get_next_lm_loader()

    model = models.MultiHeadModel(pretrained_path, with_lm, dropout)
    if load_model:
        state = torch.load(load_model, map_location="cpu")
        weights = state["model"] if "model" in state else state
        try:
            model.load_state_dict(weights)
        except Exception as e:
            print(
                colored(
                    f"fload {load_model} error, try to load with non strict mode, error is: {e}",
                    "yellow",
                )
            )
            model.load_state_dict(weights, strict=False)

    model = idist.auto_model(model)

    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)
    optimizer = idist.auto_optim(optimizer)

    scaler = torch.cuda.amp.GradScaler()
    rank_criterion = torch.nn.L1Loss().to(device)
    cls_criterion = torch.nn.BCEWithLogitsLoss().to(device)

    def train(engine, batch):
        nonlocal pair_lm_loader
        model.train()
        ids, stride_ids, mask, targets, cell_numbers = [
            item.to(device) for item in batch[:5]
        ]

        with torch.cuda.amp.autocast(enabled=True):
            in_split, rank, lm_logits = model(ids, mask)
            if with_lm:
                lm_loss = F.cross_entropy(
                    lm_logits.view(-1, vocab_len), stride_ids.view(-1)
                )
            else:
                lm_loss = torch.tensor(0)

            cls_loss = cls_criterion(in_split.squeeze(1), targets[:, 0])
            valid_ranks = targets[:, 0] == 1
            if valid_ranks.sum().item() == 0:
                rank_loss = rank_criterion(rank[0:1].squeeze(1), targets[0:1, 1])
            else:
                rank_loss = rank_criterion(
                    rank[valid_ranks].squeeze(1), targets[valid_ranks, 1]
                )
            loss = cls_loss + rank_loss + lm_loss
            scaler.scale(loss).backward()

            if pair_lm:
                try:
                    pair_lm_input_ids, pair_lm_mask_ids = next(pair_lm_loader)
                except StopIteration:
                    pair_lm_loader = get_next_lm_loader()
                    pair_lm_input_ids, pair_lm_mask_ids = next(pair_lm_loader)

                pair_lm_input_ids = pair_lm_input_ids.to(device)
                pair_lm_mask_ids = pair_lm_mask_ids.to(device)

                pair_lm_logits = model(
                    pair_lm_input_ids, pair_lm_mask_ids, True
                )
                pair_lm_loss = F.cross_entropy(
                    pair_lm_logits[:-1].view(-1, vocab_len), pair_lm_input_ids[1:].view(-1)
                )
                scaler.scale(pair_lm_loss).backward()
            else:
                pair_lm_loss = torch.tensor(0)

            if engine.state.iteration % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        return (
            loss.detach().item(),
            cls_loss.detach().item(),
            rank_loss.detach().item(),
            lm_loss.detach().item(),
            pair_lm_loss.detach().item(),
        )

    @torch.no_grad()
    def rank_eval(engine, batch):
        model.eval()
        ids, stride_ids, mask, targets, cell_numbers = [
            item.to(device) for item in batch[:5]
        ]
        sample_ids, cell_keys, split_ids = batch[5:]

        in_split, rank, _ = model(ids, mask)
        cls_loss = cls_criterion(in_split.squeeze(1), targets[:, 0])
        valid_ranks = targets[:, 0] == 1
        rank_loss = rank_criterion(
            rank[valid_ranks].squeeze(1), targets[valid_ranks, 1]
        )
        loss = cls_loss + rank_loss

        return loss, in_split, rank, sample_ids, cell_keys, split_ids

    trainer = Engine(train)
    evaluator = Engine(rank_eval)

    # trainer plugins
    # if rank == 0:
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "cls_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "rank_loss")
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, "lm_loss")
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, "pair_lm_loss")
    if rank == 0:
        ProgressBar().attach(trainer, ["loss", "cls_loss", "rank_loss", "lm_loss", "pair_lm_loss"])
        ProgressBar().attach(evaluator)

    metrics.KendallTauWithSplits(val_data, split_len).attach(evaluator, "kendall_tau")

    # scheduler
    if with_scheduler:
        train_loader = get_next_loader()
        reset_fold_idx()
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=len(train_loader) * max_epochs,
            pct_start=0.01,
            final_div_factor=10,
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def step_scheduler(engine):
            scheduler.step()

    # checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "params": params,
    }

    if with_scheduler:
        objects_to_checkpoint["lr_scheduler"] = scheduler

    checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(code_dir, require_empty=False),
        n_saved=3,
        score_name="kendall_tau",
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    if rank == 0 and not testing and saving_checkpoint:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    if resume:
        Checkpoint.load_objects(objects_to_checkpoint, torch.load(resume))

    @trainer.on(Events.EPOCH_COMPLETED(every=evaluate_every))
    def _evaluate_loss(engine: Engine):
        evaluator.run(val_loader)

    @trainer.on(Events.COMPLETED)
    def _evaluate_loss(engine: Engine):
        if engine.state.epoch % evaluate_every != 0:
            evaluator.run(val_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _replace_dataloader(engine):
        loader = get_next_loader()
        engine.state.dataloader = loader
        engine.state.epoch_length = len(loader)
        engine._setup_dataloader_iter()

    @trainer.on(Events.EPOCH_COMPLETED)
    def _testing_quit(engine):
        if testing:
            exit(0)

    trainer.run(get_next_loader(), max_epochs=max_epochs)
    idist.finalize()


def spawn(local_rank):
    fire.Fire(main)

if __name__ == "__main__":
    with idist.Parallel(backend="nccl") as parallel:
        parallel.run(spawn)

