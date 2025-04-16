import time
import torch
import hydra
import random
import logging
import transformers
from typing import Tuple
from collections import deque
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, IterableDataset
from tokenizers.processors import TemplateProcessing
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, TrainingArguments, set_seed
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers import get_cosine_schedule_with_warmup, get_wsd_schedule
from .optimizers.muon import Muon


class TokenBuffer:
    def __init__(self):
        self.buffer = deque()
    
    def add(self, tokens):
        self.buffer.extend(tokens)

    def get(self):
        tokens = list(self.buffer)
        self.buffer.clear()
        return tokens


def setup_logging() -> logging.Logger:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if rank != 0:
        logger.addFilter(lambda record: record.levelno >= logging.INFO)
    return logger


def group_tokens_into_chunks(tokenized_data, sequence_length, token_buffer: TokenBuffer):
    tokens = token_buffer.get()
    for sample in tokenized_data:
        tokens.extend(sample)
    
    total_tokens = len(tokens)
    num_full_chunks = total_tokens // sequence_length
    truncated_tokens = tokens[num_full_chunks * sequence_length:]
    token_buffer.add(truncated_tokens)
    chunk_tokens = tokens[:num_full_chunks * sequence_length]
    chunk_tokens = [chunk_tokens[i:i+sequence_length] for i in range(0, len(chunk_tokens), sequence_length)]
    return {"input_ids": chunk_tokens}


def model_provider(model_args, tokenizer: AutoTokenizer) -> LlamaForCausalLM:
    # model_args.vocab_size = tokenizer.vocab_size
    model_args.bos_token_id = tokenizer.bos_token_id
    model_args.eos_token_id = tokenizer.eos_token_id
    config = LlamaConfig(**model_args)
    model = LlamaForCausalLM(config)
    return model


def safe_load_dataset(dataset_args, retries: int = 5, base_delay: float = 1.0, backoff_factor: float = 2.0, max_delay: float = 60.0):
    for attempt in range(retries):
        try:
            return load_dataset(**dataset_args)
        except Exception as e:
            wait_time = min(base_delay * (backoff_factor ** attempt), max_delay)
            jitter = random.uniform(0, 1)
            total_wait_time = wait_time + jitter
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_wait_time:.2f} seconds...")
            time.sleep(total_wait_time)
            
    raise RuntimeError(f"Failed to load dataset after {retries} attempts. Please check your dataset configuration.")

    
def dataset_provider(task_args, max_steps: int, seed: int) -> Tuple[IterableDataset, AutoTokenizer]:
    dataset: IterableDataset = safe_load_dataset(task_args.dataset) # by default, use streaming mode
    tokenizer = AutoTokenizer.from_pretrained(task_args.tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id), 
            (f"{eos}", tokenizer.eos_token_id)
        ],
    ) 
    tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset.map(
        lambda x: tokenizer(x['text']),
        batched=True,
        remove_columns=dataset.column_names
    )
    dataset = dataset.remove_columns("attention_mask")
    dataset.shuffle(seed=seed, buffer_size=max_steps)
    token_buffer = TokenBuffer()
    dataset = dataset.map(
        lambda x: group_tokens_into_chunks(x['input_ids'], task_args.sequence_length, token_buffer),
        batched=True,
        batch_size=1024,
    )
    return dataset, tokenizer

    
def optimizer_provider(optim_args, model) -> Tuple[Optimizer, LambdaLR]:
    if optim_args.type.startswith("adamw"):
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=optim_args.learning_rate,
            betas=(optim_args.adam_beta1, optim_args.adam_beta2),
            eps=optim_args.adam_epsilon,
            weight_decay=optim_args.weight_decay,
            fused="fused" in optim_args.type,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=optim_args.warmup_steps,
            num_training_steps=optim_args.max_steps,
        )
    elif optim_args.type == "muon":
        muon_params = [p for name, p in model.named_parameters() if p.dim() >= 2 and "embed_tokens" not in name and "lm_head" not in name]
        adamw_params = [p for name, p in model.named_parameters() if p.dim() < 2 or "embed_tokens" in name or "lm_head" in name]
        optimizer = Muon(
            lr=optim_args.learning_rate,
            wd=optim_args.weight_decay,
            muon_params=muon_params,
            momentum=optim_args.momentum,
            adamw_params=adamw_params,
            adamw_betas=(optim_args.adam_beta1, optim_args.adam_beta2),
            adamw_eps=optim_args.adam_epsilon,
        )
        scheduler = get_wsd_schedule(
            optimizer,
            num_warmup_steps=optim_args.warmup_steps,
            num_decay_steps=optim_args.max_steps,
            num_training_steps=optim_args.max_steps,
            decay_type=optim_args.decay_type,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_args.optim}")
    
    return (optimizer, scheduler)


@hydra.main(version_base="1.2.0")
def train(cfg: DictConfig):
    set_seed(cfg.trainer.seed)
    logger = setup_logging()
    logger.info(f"SETTING UP MODEL AND DATASET")
    dataset, tokenizer = dataset_provider(cfg.task, cfg.trainer.max_steps, cfg.trainer.seed)
    model = model_provider(cfg.model, tokenizer)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TRAINABLE PARAMETERS: {trainable_params / 1e9:.4f}B")

    trainer = transformers.Trainer(
        model=model,
        args=TrainingArguments(**cfg.trainer),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset,
        optimizers=optimizer_provider(cfg.optimizer, model)
    )

    logger.info(f"STARTING TRAINING")
    trainer.train(resume_from_checkpoint=cfg.trainer.resume_from_checkpoint)
    trainer.save_model(cfg.trainer.output_dir)
    trainer.save_state()


if __name__ == '__main__':
    train()