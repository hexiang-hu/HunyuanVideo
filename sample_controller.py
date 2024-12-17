import os
import sys
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

import subprocess


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load the hostname from the environment variable
    hostname = os.getenv('WORKRS_HOSTNAME')
    if not hostname:
        raise EnvironmentError("Environment variable 'WORKRS_HOSTNAME' not set.")

    # Construct the input file path
    input_file_path = f"/data/hh/hunyuan_prompt/prompt-{hostname}.txt"

    # Check if the file exists
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"Input file '{input_file_path}' does not exist.")
    # Open the file and process each line

    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as fd:
            prompts = [line.strip() for line in fd.readlines()]
    else:
        raise ValueError(f'Prompt file invalid: {args.prompt_file}')

    for prompt in prompts:
        prompt_content = prompt.strip()  # Remove any extra whitespace/newlines
        if not prompt_content:  # Skip empty lines
            continue

        # Construct the system command
        command = f"torchrun --nproc_per_node=8 sample_video.py --video-size 1280 720 --video-length 129 --infer-steps 50 c--prompt='{prompt_content}' --flow-reverse --seed 42 --ulysses-degree 8 --ring-degree 1"

        print(f"Executing: {command}")

        # Execute the command
        # try:
        #     subprocess.run(command, shell=True, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error executing command: {e}")

# Example usage
if __name__ == "__main__":
    main()
