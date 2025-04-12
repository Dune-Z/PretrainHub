export HF_TOKEN=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
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
export WANDB_PROJECT=pretrain-hub
CHECKPOINT_PATH=checkpoints
RUN_NAME=50M-AdamW-LR4e-3-WM1000-STEP200000-BZ256-SEQ4096

wandb login --relogin $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 torchrun -m --nnodes=1 --nproc_per_node=8 --master_port=29501 src.train \
    --config-path ../recipes/runs \
    --config-name debug \
    trainer.deepspeed=recipes/deepspeed/zero3.json \
    trainer.output_dir="$CHECKPOINT_PATH/$RUN_NAME" \
    trainer.run_name="$RUN_NAME"

export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
for checkpoint in $CHECKPOINT_PATH/$RUN_NAME/*; do
    if [ -d "$checkpoint" ]; then
        if [ ! -f "${checkpoint}/pytorch_model.bin" ]; then
            echo "Converting $checkpoint to fp32"
            python "${checkpoint}/zero_to_fp32.py" \
                ${checkpoint} \
                ${checkpoint}
        fi
        echo "Pushing $checkpoint to hub"
        python src/push_to_hub.py \
            --model_path $checkpoint \
            --hub_model_id "YifeiZuo/$RUN_NAME"
    fi
done
