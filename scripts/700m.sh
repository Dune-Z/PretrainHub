set -x
bash install.sh
source pretrain-env/bin/activate
export HF_TOKEN=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
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
export WANDB_PROJECT=pretrain-hub
wandb login --relogin $WANDB_API_KEY

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun -m --nnodes=1 --nproc_per_node=8 --master_port=$PORT1 src.train \
    --config-path ../recipes/runs \
    --config-name 700m \
    trainer.deepspeed=recipes/deepspeed/zero3.json