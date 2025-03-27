import torch
import hydra
import transformers
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict
from datasets import load_dataset, IterableDataset
from transformers import HfArgumentParser, DataCollatorForLanguageModeling
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM


class TrainingArguments(transformers.TrainingArguments):
    pass


def model_provider(model_args):
    config = asdict(model_args)
    config = LlamaConfig(**config)
    model = LlamaForCausalLM(config)
    return model

    
def group_tokens_into_chunks(tokenized_data, sequence_length: int, batch_size: int):
    pass

    
def data_provider(task_args, tokenizer, max_steps: int):
    dataset: IterableDataset = load_dataset(**task_args.dataset) # by default, use streaming mode
    dataset.shuffle(seed=task_args.seed, buffer_size=task_args.sequence_length)
    dataset = dataset.map(lambda x: tokenizer(x['text'])['input_ids'], batched=True, remove_columns=dataset.column_names)


@hydra.main(version_base="1.2.0", config_path="recipes")
def train(cfg: DictConfig):
    pass


if __name__ == '__main__':
    train()