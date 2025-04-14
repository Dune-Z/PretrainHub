import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from types import SimpleNamespace
from train_embed import dataset_provider
from transformers import get_cosine_schedule_with_warmup
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})


def main(
    hidden_dim: int = 1024,
    num_attention_heads: int = 8,
    num_hidden_layers: int = 4,
    sequence_length: int = 2048,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    dataset_name: str = "mlfoundations/dclm-baseline-1.0",
    max_steps: int = 80000,
    batch_size: int = 32,
    learning_rate: float = 4e-3,
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    warmup_steps: int = 1000,
    seed: int = 42,
):
    task_args = {
        "tokenizer_name": tokenizer_name,
        "sequence_length": sequence_length,
        "dataset": {
            "path": dataset_name,
            "split": "train",
            "streaming": True,
        },
    }
    task_args = SimpleNamespace(**task_args)
    wandb.init(
        project="pretrain-hub",
        name="baseline",
        config={
            "hidden_dim": hidden_dim,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "sequence_length": sequence_length,
            "tokenizer_name": tokenizer_name,
            "dataset_name": dataset_name,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "warmup_steps": warmup_steps,
            "seed": seed,
        },
    )
    dataset, tokenizer = dataset_provider(
        task_args=task_args,
        max_steps=max_steps,
        seed=seed,
    )
    vocab_size = tokenizer.vocab_size
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_dim,
        intermediate_size=hidden_dim*4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=sequence_length,
        use_cache=False,
        _attn_implementation="flash_attention_2",
    )
    model = LlamaForCausalLM(config=config).to('cuda').to(torch.bfloat16)
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=1e-15,
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    progress_bar = tqdm(range(max_steps), desc="Training", unit="step")
    train_losses = []
    gradient_norms = []
    for step, sample in enumerate(dataset.iter(batch_size)):
        if step >= max_steps:
            break
        input_ids = sample["input_ids"]
        input_ids = torch.tensor(input_ids).to('cuda')
        labels = input_ids.clone()
        loss = model(input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())
        progress_bar.update(1)
        gradient_norms.append(grad_norm.item())
        train_losses.append(loss.item())
        data = {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/global_step": step,
        }
        wandb.log(data)

    wandb.finish()
    plt.figure(figsize=(12, 9))
    plt.plot(train_losses, label="Train Loss")
    plt.title("Train Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/simple_embed_train_loss.png")
    

if __name__ == "__main__":
    main()