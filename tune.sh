#/bin/bash

RUN_DIR="/home/featurize/work/ai4code/_runs/"`git rev-parse HEAD`

if git diff-index --quiet HEAD; then
  if [ ! -d RUN_DIR ]; then
    git clone /home/featurize/work/ai4code $RUN_DIR
  fi
else
  echo "Working directory is not clean, submit the changes then do the training";
  exit
fi

run_command() {
  command="$@ --git_commits `git rev-parse HEAD`"
  echo "`tput setaf 3`$command`tput sgr0`"
  $command
}

if [ $machine_name = "main" ];then
  echo "in main"
  run_command python train.py  \
      --pretrained_path /home/featurize/distilbert-base-uncased/distilbert-base-uncased  \
      --code new-base \
      --override \
      --lr 0.00005  \
      --batch_size 32 \
      --max_epochs 3  \
      --optimizer Adam \
      --context_cells_token_size 28 \
      --context_stride 1 \
      --cell_token_size 64 \
      --dataset_suffix v2
fi

if [ $machine_name = "V100_1" ];then
  echo "in V100_1"
  run_command python train.py  \
      --pretrained_path /home/featurize/distilbert-base-uncased/distilbert-base-uncased  \
      --code new-base \
      --override \
      --lr 0.00005  \
      --batch_size 32 \
      --max_epochs 3  \
      --optimizer Adam \
      --context_cells_token_size 14 \
      --context_stride 2 \
      --cell_token_size 64 \
      --dataset_suffix v2

  run_command python train.py  \
      --pretrained_path /home/featurize/distilbert-base-uncased/distilbert-base-uncased  \
      --code new-base \
      --override \
      --lr 0.00005  \
      --batch_size 32 \
      --max_epochs 3  \
      --optimizer Adam \
      --context_cells_token_size 14 \
      --context_stride 1 \
      --cell_token_size 64 \
      --dataset_suffix v2
fi

if [ $machine_name = "A6000_1" ];then
  echo "in A6000_1"

  CELL_TOKEN_SIZE=( 64 128 192 )
  CONTEXT_STRIDE=( 2 3 )
  CONTEXT_CELLS_TOKEN_SIZE=( 18 )

  for cell_token_size in "${CELL_TOKEN_SIZE[@]}"; do
    for context_stride in "${CONTEXT_STRIDE[@]}"; do
      for context_cells_token_size in "${CONTEXT_CELLS_TOKEN_SIZE[@]}"; do
        cd $RUN_DIR && run_command \
          python train.py  \
            --pretrained_path /home/featurize/distilbert-base-uncased/distilbert-base-uncased  \
            --code tune_params \
            --override \
            --lr 0.00005  \
            --batch_size 128 \
            --max_epochs 2  \
            --optimizer Adam \
            --context_cells_token_size $context_cells_token_size \
            --context_stride $context_stride \
            --cell_token_size $cell_token_size \
            --dataset_suffix v2
      done
    done
  done
fi
