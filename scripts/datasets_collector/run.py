import argparse
import bittensor as bt
from huggingface_hub import login
from metadata import load_metadata_info
from wandb_utils import collect_wandb_data
from openai_dataset_collector import extract_openai_data, OPENAI_DATASET_PATH
from openai_dataset_collector import DEFAULT_HF_DATASET_OUTPUT_DIR as OPENAI_HF_DATASET_OUTPUT_DIR
from datasets import disable_progress_bar

SUPPORTED_VERSIONS = ['1.0.0', '1.0.1', '1.0.2', '1.0.3', '1.0.4', '1.0.5', '1.0.6']

DEFAULT_HF_DATASET_OUTPUT_DIR = 'pedroferreira/openvalidators'
DEFAULT_WANDB_PROJECT = 'opentensor-dev/openvalidators'
HF_TOKEN = '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'


def start_collector(version: str, hf_dataset_output_dir: str, wandb_project: str):
    """Starts the data collector script to extract data from wandb into Hugging Face datasets
    Args:
        version (str): Version of the dataset to collect
        hf_dataset_output_dir (str): Hugging Face dataset output directory
        wandb_project (str): Wandb project to crawl
    """
    bt.logging.info(f"OpenValidators dataset: Starting openvalidators dataset collector for version: {version}")
    # Load metadata info
    metadata_info = load_metadata_info(hf_datasets_path=hf_dataset_output_dir, version=version)

    # Collecting wandb data
    output_result = collect_wandb_data(
        metadata_info_df=metadata_info,
        wandb_project=wandb_project,
        version=version,
        hf_datasets_path=hf_dataset_output_dir
    )

    new_data_found = (output_result.new_downloaded_run_ids > 0 or
                      output_result.skipped_run_ids > 0 or
                      len(output_result.problematic_runs) > 0)

    if new_data_found:
        bt.logging.success(f"Runs from version {version} collected successfully!")
        bt.logging.info(f"New downloaded runs: {output_result.new_downloaded_run_ids}")
        bt.logging.info(f"Skipped runs: {output_result.skipped_run_ids}")

        problematic_run_ids = [run.run_id for run in output_result.problematic_runs]

        if len(problematic_run_ids) > 0:
            bt.logging.warning(f"Problematic runs({len(output_result.problematic_runs)}): {problematic_run_ids}")

    bt.logging.info('*' * 100)


if __name__ == "__main__":
    """Executes the data collector script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect data from wandb and save to huggingface datasets")
    parser.add_argument("--wandb_project", type=str, help="Wandb project to crawl", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--hf_dataset_output_dir", type=str, help="Hugging Face dataset output directory", default=DEFAULT_HF_DATASET_OUTPUT_DIR)
    parser.add_argument("--hf_token", type=str, help="Hugging Face token", default=HF_TOKEN)
    args = parser.parse_args()

    # Disables hf progress bar
    disable_progress_bar()

    hf_token = args.hf_token
    openvalidators_hf_dataset_dir = args.hf_dataset_output_dir
    wandb_project = args.wandb_project

    # Login to hugging face
    login(hf_token)

    for version in SUPPORTED_VERSIONS:
        # Start data collector from wandb to hf datasets
        start_collector(version=version, hf_dataset_output_dir=openvalidators_hf_dataset_dir, wandb_project=wandb_project)

        # Extract data openvalidators hf datasets to openvalidators openai dataset
        bt.logging.info(f"OpenAI dataset: Extracting data from version {version}")
        extract_openai_data(
            openvalidators_version=version,
            hf_source_dataset=openvalidators_hf_dataset_dir,
            hf_dataset_output_dir=OPENAI_HF_DATASET_OUTPUT_DIR,
            openai_dataset_path=OPENAI_DATASET_PATH
        )
        bt.logging.info('*' * 100)



