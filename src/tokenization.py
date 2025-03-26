import os
from datasets import load_dataset
from transformers import AutoTokenizer


def group_into_chunks(tokenized_data, chunk_size):
    pass


def tokenize_and_save(config_card):
    # dataset = load_dataset("mlfoundations/dclm-baseline-1.0", split=config_card["split"])
    dataset = load_dataset(config_card["dataset_path"], name=config_card["subset"], split=config_card["split"])
    tokenizer = AutoTokenizer.from_pretrained(config_card["model_name"])
    num_proc = max(os.cpu_count() - 1, 1)
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples[config_card["text_column"]]),
        batched=False,
        num_proc=num_proc,
    )

    
    
if __name__ == '__main__':
    config_card = {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "dataset_path": "allenai/c4",
        "subset": "en",
        "split": "train",
        "text_column": "text",
    }
    tokenize_and_save(config_card)

