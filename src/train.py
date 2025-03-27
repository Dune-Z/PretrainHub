import torch
import hydra
import logging
import transformers
from typing import Tuple
from collections import deque
from omegaconf import DictConfig
from datasets import load_dataset, IterableDataset
from tokenizers.processors import TemplateProcessing
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, TrainingArguments, set_seed
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM


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

    
def dataset_provider(task_args, max_steps: int, seed: int) -> Tuple[IterableDataset, AutoTokenizer]:
    dataset: IterableDataset = load_dataset(**task_args.dataset) # by default, use streaming mode
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
    )

    logger.info(f"STARTING TRAINING")
    trainer.train(resume_from_checkpoint=cfg.trainer.resume_from_checkpoint)
    trainer.save_model(cfg.trainer.output_dir)
    trainer.save_state()


if __name__ == '__main__':
    train()