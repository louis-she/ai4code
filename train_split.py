import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import torch
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

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TOKENIZERS_PARALLELISM'] = "false"

LOG_DIR = Path("/home/featurize/ai4code")
DEVICE = torch.device("cuda")

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
    override: bool = False,
    optimizer: str = "AdamW",
    testing: bool = False,
    saving_checkpoint: bool = True,
    num_workers: int = 2,
    train_folds: Tuple[int] = (1,),
    val_folds: Tuple[int] = (0,),
    evaluate_every: int = 1,
    with_scheduler: bool = False,
    extra_vocab: str = None,
    ordered_context_ratio: float = 0.3,
    validate_with_ordered: bool = False,
    split_len: int = 8,
    # dataset temp
    negative_ratio: float = 0.5,
    cell_token_size: int = 64,
    cell_stride: int = 1,
    context_cells_token_size: int = 23,
    context_stride: int = 1,
    max_len: int = 512,
    train_num_samples: int = None,
    val_num_samples: int = None,
    dropout: float = 0.2,
    train_all_cells: bool = False,
):
    params = SerializableDict(locals())
    torch.manual_seed(seed)

    max_epochs = max_epochs * len(train_folds)

    if testing:
        train_num_samples = 10
        val_num_samples = 10
        code = "test_" + code
        override = True

    code_dir = LOG_DIR / code
    if code_dir.exists() and not override:
        exit(
            f"Code dir {code_dir} exists! use --override to force override it or change another code name"
        )

    data: Dict[str, datasets.Sample] = pickle.load(
        open(f"/home/featurize/work/ai4code/data/10fold{'_mini' if testing else ''}.{dataset_suffix}.pkl", "rb")
    )

    val_data = {k: v for k, v in list(data.items()) if v.fold in val_folds}
    if val_num_samples is not None:
        val_data = {k: v for k, v in list(val_data.items())[:val_num_samples]}

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path, do_lower_case=True, use_fast=True
    )
    vocab_len = len(tokenizer)

    if extra_vocab:
        extra_vocab: Counter = pickle.load(open(extra_vocab, "rb"))
        tokenizer.add_tokens(x[0] for x in extra_vocab.most_common(2000))

    special_tokens = datasets.SpecialTokenID(
        hash_id=tokenizer.encode("#", add_special_tokens=False)[0],
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    del tokenizer

    current_train_fold_idx = 0

    def reset_fold_idx():
        nonlocal current_train_fold_idx
        current_train_fold_idx = 0

    def create_dataset(data, ordered_context_ratio):
        return datasets.RankDatasetWithSplits(
            data,
            special_tokens=special_tokens,
            cell_token_size=cell_token_size,
            cell_stride=cell_stride,
            context_cells_token_size=context_cells_token_size,
            context_stride=context_stride,
            max_len=max_len,
            ordered_context_ratio=ordered_context_ratio,
            split_len=split_len,
        )

    def get_next_loader():
        # 由于内存原因，每个 DataLoader 只能遍历单个 Fold
        # 因此一轮表示单个 Fold 的训练，每次训练完毕后手动
        # 把 Loader 改为下一个 Fold
        nonlocal current_train_fold_idx
        train_data = {
            k: v
            for k, v in list(data.items())
            if v.fold == int(train_folds[current_train_fold_idx % len(train_folds)])
        }
        if train_num_samples is not None:
            train_data = {k: v for k, v in list(train_data.items())[:train_num_samples]}

        current_train_fold_idx += 1

        return DataLoader(
            create_dataset(train_data, ordered_context_ratio=ordered_context_ratio),
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
        )

    val_loader = DataLoader(
        create_dataset(val_data, ordered_context_ratio=1 if validate_with_ordered else 0),
        num_workers=num_workers,
        batch_size=batch_size,
    )

    model = models.MultiHeadModel(pretrained_path, dropout)
    if extra_vocab:
        model.backbone.resize_token_embeddings(vocab_len)
    model.to(DEVICE)

    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()
    rank_criterion = torch.nn.L1Loss()
    cls_criterion = torch.nn.BCEWithLogitsLoss()

    def train(engine, batch):
        model.train()
        ids, mask, targets, cell_numbers = [item.to(DEVICE) for item in batch[:4]]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            in_split, rank = model(ids, mask, cell_numbers)
            cls_loss = cls_criterion(in_split.squeeze(1), targets[:, 0])
            valid_ranks = targets[:, 0] == 1
            rank_loss = rank_criterion(rank[valid_ranks].squeeze(1), targets[valid_ranks, 1])
            loss = cls_loss + rank_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach().item(), cls_loss.detach().item(), rank_loss.detach().item()

    @torch.no_grad()
    def rank_eval(engine, batch):
        model.eval()
        ids, mask, targets, cell_numbers = [item.to(DEVICE) for item in batch[:4]]
        sample_ids, cell_keys, split_ids = batch[4:]

        in_split, rank = model(ids, mask, cell_numbers)
        cls_loss = cls_criterion(in_split.squeeze(1), targets[:, 0])
        valid_ranks = targets[:, 0] == 1
        rank_loss = rank_criterion(rank[valid_ranks].squeeze(1), targets[valid_ranks, 1])
        loss = cls_loss + rank_loss

        return loss, in_split, rank, sample_ids, cell_keys, split_ids

    trainer = Engine(train)
    evaluator = Engine(rank_eval)

    # trainer plugins
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "cls_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "rank_loss")
    ProgressBar().attach(trainer, ["loss", "cls_loss", "rank_loss"])

    # evaluator plugins
    ProgressBar().attach(evaluator)
    metrics.KendallTauWithSplits(val_data, split_len).attach(evaluator, "kendall_tau")

    # aim
    aim_logger = AimLogger(
        # repo="aim://172.16.190.45:53800",
        repo=os.path.join(LOG_DIR / "aim"),
        experiment=code,
    )
    aim_logger.log_params(params.state_dict())

    aim_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="train",
        output_transform=lambda x: x[0],
    )
    aim_logger.attach_output_handler(
        evaluator,
        event_name=Events.ITERATION_COMPLETED,
        tag="val",
        output_transform=lambda x: x[0],
    )
    aim_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=["kendall_tau"],
        global_step_transform=global_step_from_engine(
            trainer, Events.ITERATION_COMPLETED
        ),
    )

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
            aim_logger.log_metrics(
                {"lr": scheduler.get_last_lr()[0]}, step=engine.state.iteration
            )

    # checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "params": params
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
    if not testing and saving_checkpoint:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    if resume:
        Checkpoint.load_objects(objects_to_checkpoint, torch.load(resume))

    @trainer.on(Events.EPOCH_COMPLETED)
    def _replace_dataloader(engine):
        loader = get_next_loader()
        engine.state.dataloader = loader
        engine.state.epoch_length = len(loader)
        engine._setup_dataloader_iter()

    @trainer.on(Events.EPOCH_COMPLETED(every=evaluate_every))
    def _evaluate_loss(engine):
        evaluator.run(val_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _testing_quit(engine):
        if testing:
            exit(0)

    trainer.run(get_next_loader(), max_epochs=max_epochs)


if __name__ == "__main__":
    fire.Fire(main)
