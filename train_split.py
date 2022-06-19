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
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import RunningAverage
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer

from ai4code import datasets, metrics, models, utils
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
    split_len: int = 8,
    accumulation_steps: int = 1,
    pair_lm: bool = False,
    # dataset temp

    anchor_size: int = 64,
    max_len: int = 256,
    train_num_samples: int = None,
    val_num_samples: int = None,
    dropout: float = 0.2,
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

    val_data = pickle.load(
        open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/0.pkl", "rb")
    )
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

    def create_dataset(data, only_task_data=False):
        return datasets.MixedDatasetWithSplits(
            data,
            special_tokens=special_tokens,
            anchor_size=anchor_size,
            max_len=max_len,
            split_len=split_len,
            distil_context=distil_context,
            only_task_data=only_task_data
        )

    def get_next_loader():
        nonlocal current_train_fold_idx
        fold = train_folds[current_train_fold_idx % len(train_folds)]
        if rank == 0:
            print(colored(f"generate loader of fold {fold}", "green"))
        train_data = pickle.load(
            open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/{fold}.pkl", "rb")
        )
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
        lm_train_data = pickle.load(
            open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/{fold}.pkl", "rb")
        )
        current_lm_fold_idx += 1

        return iter(
            idist.auto_dataloader(
                datasets.PairLMDataset(lm_train_data, special_tokens, max_len),
                batch_size=batch_size,
                shuffle=True,
            )
        )

    val_loader = idist.auto_dataloader(
        create_dataset(val_data, only_task_data=True),
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if pair_lm:
        lm_loader = get_next_lm_loader()

    model = models.MultiHeadModel(pretrained_path, pair_lm, dropout)
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
        nonlocal lm_loader
        model.train()
        input_ids, mask, lm_input_ids, lm_mask, targets, lm_targets = [
            item.to(device) for item in batch[:6]
        ]

        real_bs = input_ids.shape[0]
        # input_ids, mask, lm_input_ids, lm_mask: bs x 256
        input_ids = torch.cat([input_ids, lm_input_ids], dim=0)
        mask = torch.cat([mask, lm_mask], dim=0)
        assert input_ids.shape[0] == mask.shape[0]

        shuffle_indices, unshuffle_indices = utils.shuffle_batch(input_ids)
        input_ids = input_ids[shuffle_indices]
        mask = mask[shuffle_indices]

        with torch.cuda.amp.autocast(enabled=True):
            in_split, rank, lm_logits = model(input_ids, mask)

            in_split = in_split[unshuffle_indices][:real_bs]
            rank = rank[unshuffle_indices][:real_bs]
            lm_logits = lm_logits[unshuffle_indices][real_bs:]

            split_loss = cls_criterion(in_split.squeeze(1), targets[:, 0])

            valid_ranks = targets[:, 0] == 1
            if valid_ranks.sum().item() == 0:
                rank_loss = rank_criterion(rank[0:1].squeeze(1), targets[0:1, 1])
            else:
                rank_loss = rank_criterion(
                    rank[valid_ranks].squeeze(1), targets[valid_ranks, 1]
                )

            lm_loss = F.binary_cross_entropy_with_logits(lm_logits, lm_targets)
            loss = split_loss + rank_loss + lm_loss

            scaler.scale(loss).backward()
            if engine.state.iteration % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        return (
            loss.detach().item(),
            split_loss.detach().item(),
            rank_loss.detach().item(),
            lm_loss.detach().item(),
        )

    @torch.no_grad()
    def rank_eval(engine, batch):
        model.eval()
        input_ids, mask, targets = [item.to(device) for item in batch[:3]]
        sample_ids, cell_keys, split_ids = batch[3:]

        in_split, rank = model(input_ids, mask, lm=False)
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
    if rank == 0:

        @trainer.on(Events.ITERATION_COMPLETED(every=50))
        def _print_progress(engine):
            print(
                "({}/{}@{}) loss: {:10.4f}\tcls_loss: {:10.4f}\trank_loss: {:10.4f}\tlm_loss: {:10.4f}\t"
            .format(
                engine.state.iteration % engine.state.epoch_length,
                engine.state.epoch_length,
                engine.state.epoch,
                engine.state.metrics["loss"],
                engine.state.metrics["cls_loss"],
                engine.state.metrics["rank_loss"],
                engine.state.metrics["lm_loss"],
            ))

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
