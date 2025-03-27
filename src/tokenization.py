import os
from collections import deque
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing


class TokenBuffer:
    def __init__(self):
        self.buffer = deque()
    
    def add(self, tokens):
        self.buffer.extend(tokens)

    def get(self):
        tokens = list(self.buffer)
        self.buffer.clear()
        return tokens


def group_tokens_into_chunks(tokenized_data, sequence_length, token_buffer: TokenBuffer):
    tokens = token_buffer.get()
    for sample in tokenized_data:
        tokens.extend(sample)
    
    total_tokens = len(tokens)
    num_full_chunks = total_tokens // sequence_length
    truncated_tokens = tokens[:num_full_chunks * sequence_length]
    token_buffer.add(truncated_tokens)
    chunk_tokens = tokens[:num_full_chunks * sequence_length]
    chunk_tokens = [chunk_tokens[i:i+sequence_length] for i in range(0, len(chunk_tokens), sequence_length)]
    return {"input_ids": chunk_tokens}


def dataset_provider(config_card, sequence_length=4096):
    dataset = load_dataset(
        config_card["dataset_path"],
        name=config_card.get("subset", None),
        split=config_card.get("split", None),
        streaming=config_card.get("streaming", False),
    )
    tokenizer = AutoTokenizer.from_pretrained(config_card["model_name"])
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
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples[config_card["text_column"]]),
        batched=False,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset = tokenized_dataset.remove_columns("attention_mask")
    token_buffer = TokenBuffer()
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: group_tokens_into_chunks(examples['input_ids'], sequence_length, token_buffer),
        batched=True,
    )
    return tokenized_dataset
    
    
if __name__ == '__main__':
    config_card = {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "dataset_path": "mlfoundations/dclm-baseline-1.0",
        "split": "train",
        "text_column": "text",
        "streaming": True
    }
    dataset = dataset_provider(config_card)
    for sample in dataset.iter(batch_size=128):
        print(sample.keys())
        print(len(sample["input_ids"]))
        print(len(sample["input_ids"][0]))
        breakpoint()

