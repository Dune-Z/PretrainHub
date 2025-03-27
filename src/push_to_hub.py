import argparse
from transformers import AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/debug")
    parser.add_argument("--hub_model_id", type=str, default="Ethan-Z/debug")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.push_to_hub(args.hub_model_id)