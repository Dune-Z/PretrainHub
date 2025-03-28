set -x
bash install.sh
source pretrain-env/bin/activate
export HF_TOKEN=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
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
export WANDB_PROJECT=pretrain-hub
wandb login --relogin $WANDB_API_KEY

torchrun -m --nnodes=4 --nproc_per_node=8 --master_port=$PORT1 src.train \
    --config-path ../recipes/runs \
    --config-name 1.5B \
    trainer.deepspeed=recipes/deepspeed/zero3.json \
    trainer.gradient_accumulation_steps=2 \
    trainer.max_steps=50000
