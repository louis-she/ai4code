"""
作为 Rank 之后的精修后处理。

训练部分：预测两个 markdown 的「距离」
"""
import gc
import os
import pickle
from pathlib import Path
import random
from typing import Tuple
from termcolor import colored
from ignite.contrib.handlers import wandb_logger

import fire
import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import RunningAverage
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

import ai4code
from ai4code import datasets, metrics, models, utils
from ai4code.utils import SerializableDict
import ignite.distributed as idist
from torch.optim.swa_utils import AveragedModel
from transformers import logging
from ignite.metrics import Precision, Recall, Accuracy


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
    train_folds: Tuple[str] = ("1",),
    val_folds: Tuple[int] = (0,),
    evaluate_every: int = 30000,
    accumulation_steps: int = 1,
    max_len: int = 256,
    train_num_samples: int = None,
    val_num_samples: int = None,
    do_evaluation: bool = False,
):
    rank = idist.get_local_rank()
    params = SerializableDict(locals())
    if rank == 0:
        utils.print_params(params.state_dict())
    torch.manual_seed(seed)
    device = idist.device()

    train_folds = train_folds.split(",")
    max_epochs = max_epochs * len(train_folds)
    random.shuffle(train_folds)

    if testing:
        train_num_samples = 20
        val_num_samples = 20
        code = "test_" + code
        override = True

    code_dir = LOG_DIR / code
    if code_dir.exists() and not override:
        exit(
            f"Code dir {code_dir} exists! use --override to force override it or change another code name"
        )

    val_data = {}
    for i in val_folds:
        val_data = {
            **pickle.load(
                open(f"/home/featurize/work/ai4code/data/{dataset_suffix}/{i}.pkl", "rb")
            ),
            **val_data,
        }

    if val_num_samples is not None:
        val_data = {k: v for k, v in list(val_data.items())[:val_num_samples]}

    special_tokens = datasets.SpecialTokenID(
        **pickle.load(
            open(
                f"/home/featurize/work/ai4code/data/{dataset_suffix}/special_tokens.pkl",
                "rb",
            )
        )
    )

    current_train_fold_idx = 0

    def create_dataset(data, val=False):
        return datasets.PairwiseDataset(
            data,
            special_tokens=special_tokens,
            max_len=max_len,
            val=val
        )

    def get_next_loader():
        nonlocal current_train_fold_idx
        fold = train_folds[current_train_fold_idx % len(train_folds)]
        if rank == 0:
            print(colored(f"generate fold data {fold}", "green"))
        with open(f"/home/featurize/work/ai4code/data/{fold}", "rb") as f:
            train_data = pickle.load(f)
        if train_num_samples is not None:
            train_data = {k: v for k, v in list(train_data.items())[:train_num_samples]}

        current_train_fold_idx += 1

        return idist.auto_dataloader(
            create_dataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )

    val_loader = idist.auto_dataloader(
        create_dataset(val_data, val=True),
        num_workers=num_workers,
        batch_size=batch_size,
    )
    model = models.PairwiseModel(pretrained_path)

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
            try:
                model.load_state_dict(weights, strict=False)
            except Exception as e:
                print(colored(f"Still a error raised: {e}, will try to load only the transformers weights", "red"))
                weights = {k: v for k, v in weights.items() if k.startswith("backbone")}
                model.load_state_dict(weights, strict=False)
        del state
        del weights

    model = idist.auto_model(model)

    optimizer = getattr(optim, optimizer)(model.parameters(), lr=lr)
    optimizer = idist.auto_optim(optimizer)

    scaler = torch.cuda.amp.GradScaler()
    cls_criterion = torch.nn.BCEWithLogitsLoss().to(device)

    def train(engine, batch):
        model.train()
        input_ids, mask, targets = [item.to(device) for item in batch]
        with torch.cuda.amp.autocast(enabled=True):
            preds = model(input_ids, mask)
            loss = cls_criterion(preds, targets)
        scaler.scale(loss).backward()

        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return loss

    @torch.no_grad()
    def rank_eval(engine, batch):
        model.eval()
        input_ids, mask, targets = [
            item.to(device) for item in batch
        ]
        preds = model(input_ids, mask)
        return preds > 0.5, targets

    trainer = Engine(train)
    evaluator = Engine(rank_eval)

    metric = Accuracy()
    metric.attach(evaluator, 'accuracy')

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    if rank == 0:
        @trainer.on(Events.ITERATION_COMPLETED(every=50))
        def _print_progress(engine):
            print(
                "({}/{}@{}) loss: {:10.4f}".format(
                    engine.state.iteration % engine.state.epoch_length,
                    engine.state.epoch_length,
                    engine.state.epoch,
                    engine.state.metrics["loss"],
                )
            )

    # checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "params": params,
    }

    checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(code_dir, require_empty=False),
        n_saved=3,
        score_name="accuracy",
        global_step_transform=lambda *_: trainer.state.iteration,
    )
    if rank == 0 and not testing and saving_checkpoint:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    if resume:
        Checkpoint.load_objects(objects_to_checkpoint, torch.load(resume))

    @trainer.on(Events.ITERATION_COMPLETED(every=evaluate_every))
    def _evaluate_loss(engine: Engine):
        evaluator.run(val_loader)

    @trainer.on(Events.COMPLETED)
    def _evaluate_loss(engine: Engine):
        logging.info("training ended, now run one last evaluation")
        evaluator.run(val_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _replace_dataloader(engine):
        loader = get_next_loader()
        engine.set_data(loader)
        engine.state.epoch_length = len(loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _testing_quit(engine):
        if testing:
            exit(0)

    wandb = None
    if rank == 0 and not testing and len(git_commit) == 40:
        wandb = wandb_logger.WandBLogger(
            project="ai4code",
            entity="chenglu",
            name=code,
            config=params,
        )

        wandb.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"loss": loss[0]}
        )

        wandb.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["kendall_tau"],
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _print_result(engine):
        print(engine.state.metrics["accuracy"])

    if not do_evaluation:
        trainer.run(get_next_loader(), max_epochs=max_epochs)
    else:
        evaluator.run(val_loader)

    idist.finalize()
    if rank == 0 and wandb:
        wandb.close()


def spawn(local_rank):
    fire.Fire(main)


if __name__ == "__main__":
    with idist.Parallel(backend="nccl") as parallel:
        parallel.run(spawn)
