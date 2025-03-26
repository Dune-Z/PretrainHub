import torch
import hydra
import transformers
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict
from transformers import HfArgumentParser
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM


def model_provider(model_args):
    config = asdict(model_args)
    config = LlamaConfig(**config)
    model = LlamaForCausalLM(config)
    return model

    
def data_provider(task_args):
    pass


@hydra.main(version_base="1.2.0", config_path="recipes")
def train(cfg: DictConfig):
    pass


if __name__ == '__main__':
    train()