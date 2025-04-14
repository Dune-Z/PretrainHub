import argparse
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/debug")
    parser.add_argument("--hub_model_id", type=str, default="Ethan-Z/debug")
    parser.add_argument("--wandb_logging", action="store_true", help="Whether to use wandb logging")
    args = parser.parse_args()
    if args.wandb_logging:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=args.model_path,
            path_in_repo="wandb.zip",
            repo_id=args.hub_model_id,
            repo_type="model",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.push_to_hub(args.hub_model_id)