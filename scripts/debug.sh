export HF_TOKEN=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
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3,8,9 torchrun -m --nnodes=1 --nproc_per_node=6 src.train \
    --config-path ../recipes/runs \
    --config-name debug