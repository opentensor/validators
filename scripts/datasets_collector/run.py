import argparse
import bittensor as bt
from huggingface_hub import login
from metadata import load_metadata_info
from wandb_utils import collect_wandb_data

#DEFAULT_VERSION = openvalidators.__version__
DEFAULT_VERSION = '1.0.0'
DEFAULT_HF_DATASET_OUTPUT_DIR = 'opentensor/openvalidators-test'
DEFAULT_WANDB_PROJECT = 'opentensor-dev/openvalidators'
HF_TOKEN = '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'


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect data from wandb and save to huggingface datasets")
    parser.add_argument("--wandb_project", type=str, help="Wandb project to crawl", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--version", type=str, help="Tag version of openvalidators to be crawled", default=DEFAULT_VERSION)
    parser.add_argument("--hf_dataset_output_dir", type=str, help="Hugging Face dataset output directory", default=DEFAULT_HF_DATASET_OUTPUT_DIR)
    parser.add_argument("--hf_token", type=str, help="Hugging Face token", default=HF_TOKEN)
    args = parser.parse_args()

    # Login to hugging face
    login(args.hf_token)

    # Load metadata info
    metadata_info = load_metadata_info(hf_datasets_path=args.hf_dataset_output_dir, version=args.version)

    # Collecting wandb data
    output_result = collect_wandb_data(
        metadata_info_df=metadata_info,
        wandb_project=args.wandb_project,
        version=args.version,
        hf_datasets_path=args.hf_dataset_output_dir
    )

    bt.logging.info(f"Runs from version {args.version} collected successfully!")
    bt.logging.info(f"New downloaded runs: {output_result.new_downloaded_run_ids}")
    bt.logging.info(f"Skipped runs: {output_result.skipped_run_ids}")

    problematic_run_ids = [run.run_id for run in output_result.problematic_runs]
    bt.logging.info(f"Problematic runs({len(output_result.problematic_runs)}): {problematic_run_ids}")


