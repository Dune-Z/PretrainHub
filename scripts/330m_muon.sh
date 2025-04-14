#!/bin/bash
set -euo pipefail
set -x

usage() {
  echo "Usage: $0 --train-hf-token <TRAIN_HF_TOKEN> --push-hf-token <PUSH_HF_TOKEN> --wandb-token <WANDB_TOKEN>"
  exit 1
}

TRAIN_HF_TOKEN=""
PUSH_HF_TOKEN=""
WANDB_TOKEN=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --train-hf-token)
      TRAIN_HF_TOKEN="$2"
      shift ;;
    --push-hf-token)
      PUSH_HF_TOKEN="$2"
      shift ;;
    --wandb-token)
      WANDB_TOKEN="$2"
      shift ;;
    *) 
      echo "Unknown parameter passed: $1"
      usage ;;
  esac
  shift
done

if [[ -z "$TRAIN_HF_TOKEN" || -z "$PUSH_HF_TOKEN" || -z "$WANDB_TOKEN" ]]; then
  echo "Missing one or more required parameters."
  usage
fi

bash install.sh
source pretrain-env/bin/activate
export WANDB_PROJECT=pretrain-hub
CHECKPOINT_PATH=checkpoints
RUN_NAME=330M-Muon-LR8e-3-WM0-STEP200000-BZ256-SEQ4096
CONFIG_NAME=330m-muon

wandb login --relogin $WANDB_TOKEN
huggingface-cli login --token $TRAIN_HF_TOKEN --add-to-git-credential
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun -m --nnodes=1 --nproc_per_node=8 --master_port=$PORT1 src.train \
    --config-path ../recipes/runs \
    --config-name $CONFIG_NAME \
    trainer.deepspeed=recipes/deepspeed/zero3.json \
    trainer.output_dir="$CHECKPOINT_PATH/$RUN_NAME" \
    trainer.run_name="$RUN_NAME"

huggingface-cli login --token $PUSH_HF_TOKEN --add-to-git-credential
for checkpoint in $CHECKPOINT_PATH/$RUN_NAME/*; do
    if [ -d "$checkpoint" ] && [[ "$(basename "$checkpoint")" != "wandb" ]]; then
        if [ ! -f "${checkpoint}/pytorch_model.bin" ] && [ ! -f "${checkpoint}/model.safetensors" ]; then
            echo "Converting $checkpoint to fp32"
            python "${checkpoint}/zero_to_fp32.py" \
                ${checkpoint} \
                ${checkpoint}
        fi
        echo "Pushing $checkpoint to hub"
        checkpoint_name=$(basename "$checkpoint")
        python src/push_to_hub.py \
            --model_path $checkpoint \
            --hub_model_id "YifeiZuo/${RUN_NAME}-${checkpoint_name}"
    fi
done