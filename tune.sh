#/bin/bash

# set -x

if git diff-index --quiet HEAD; then
  mkdir _runs/`git rev-parse HEAD`
  exit;
else
  echo "Working directory is not clean, submit the changes then do the training";
  # branch_name=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 8)
  exit;
fi

if [ $machine_name = "main" ];then
  echo "in main"
  python train.py  \
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

  python train.py  \
      --pretrained_path /home/featurize/distilbert-base-uncased/distilbert-base-uncased  \
      --code new-base \
      --override \
      --lr 0.00005  \
      --batch_size 32 \
      --max_epochs 3  \
      --optimizer Adam \
      --context_cells_token_size 28 \
      --context_stride 2 \
      --cell_token_size 64 \
      --dataset_suffix v2
fi

if [ $machine_name = "V100_1" ];then
  echo "in V100_1"
  python train.py  \
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

  python train.py  \
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
  CONTEXT_STRIDE=( 1 2 3 )
  CONTEXT_CELLS_TOKEN_SIZE=( 14 28 )

  for cell_token_size in "${CELL_TOKEN_SIZE[@]}"; do
    for context_stride in "${CONTEXT_STRIDE[@]}"; do
      for context_cells_token_size in "${CONTEXT_CELLS_TOKEN_SIZE[@]}"; do
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
