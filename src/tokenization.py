import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing


def group_into_chunks(tokenized_data, chunk_size):
    pass


def tokenize_and_save(config_card):
    # dataset = load_dataset("mlfoundations/dclm-baseline-1.0", split=config_card["split"])
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
        lambda examples: tokenizer(examples[config_card["text_column"]])['input_ids'],
        batched=False,
        remove_columns=dataset.column_names,
    )
    for t in tokenized_dataset.iter(batch_size=4):
        pass
    
    
if __name__ == '__main__':
    config_card = {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "dataset_path": "allenai/c4",
        "subset": "en",
        "split": "train",
        "text_column": "text",
    }
    config_card = {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "dataset_path": "mlfoundations/dclm-baseline-1.0",
        "split": "train",
        "text_column": "text",
        "streaming": True
    }
    tokenize_and_save(config_card)

