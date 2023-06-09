import wandb
import argparse
import tqdm
import pandas as pd
import openvalidators
import bittensor as bt
from loguru import logger
from datetime import datetime
from huggingface_hub import HfFileSystem, login

DEFAULT_VERSION = openvalidators.__version__
DEFAULT_HF_DATASET_OUTPUT_DIR = 'pedroferreira/test-release/'
DEFAULT_WANDB_PROJECT = 'opentensor-dev/openvalidators'
HF_TOKEN = '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'


def load_metadata_info(hf_datasets_path: str, version: str) -> pd.DataFrame:
    metadata_path = f"datasets/{hf_datasets_path}/{version}/metadata.json"

    hf_file_system = HfFileSystem()
    repo_metadata = hf_file_system.glob(metadata_path)

    repo_metadata_exists = len(repo_metadata) != 0

    if repo_metadata_exists:
        bt.logging.info(f'Metadata file located at {metadata_path}, loading metadata file...')
        # Reads JSON file directly from Hugging Face Hub
        metadata_info_df = pd.read_json(f"hf://{metadata_path}")
        bt.logging.info(f'Metadata file loaded successfully!')

        return metadata_info_df
    else:
        bt.logging.info(f'No metadata file located at {metadata_path}, new metadata file will be created...')
        return pd.DataFrame()


def export_df_to_hf(
    run_df: pd.DataFrame,
    hf_datasets_path: str,
    version: str,
    run: wandb.Run
):
    run_id_file_output_path = f"datasets/{hf_datasets_path}/{version}/raw_data/{run.id}.json"

    bt.logging.info(f'Exporting run data to {run_id_file_output_path}...')
    # TODO: add complementary metadata to the run_df
    run_df.to_parquet(run_id_file_output_path)
    bt.logging.info(f'Run data exported successfully!')


def consume_wandb_run(
    run: wandb.Run,
    wandb_project: str,
    wandb_api: wandb.Api,
    hf_datasets_path: str,
    version: str,
    metadata_info_df: pd.DataFrame,

):
    run_id_metadata_row = metadata_info_df.loc[metadata_info_df['run_id'] == run.id]

    # Converts dataframe row to Metadata object
    run_id_metadata = run_id_metadata_row.iloc[0].to_dict()
    completed_states = ["finished", "failed", "crashed", "killed"]

    if run_id_metadata["completed"] and run_id_metadata["downloaded"]:
        # Checks if run_id is already completed and downloaded. If so, skips run_id as it is already captured
        bt.logging.info(f'Run {run.id} already completed and downloaded, skipping...')
        return
    elif run.state == "running":
        # Checks if run_id is still running. If so, skips run_id and updates last checkpoint
        bt.logging.info(f'Run {run.id} is still running, skipping and updating checkpoint...')

        # Updates last checkpoint of run_id
        metadata_info_df.loc[run_id_metadata_row, 'last_checkpoint'] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        metadata_info_df.to_json(f"hf://datasets/{hf_datasets_path}/{version}/metadata.json")
        return
    elif run.state in completed_states and not run_id_metadata["downloaded"]:
        # Checks if run_id is already completed but not downloaded. If so, downloads run_id and updates metadata
        bt.logging.info(f'Run {run.id} already completed with state {run.state}, downloading it...')

        run = wandb_api.run(f'{wandb_project}/{run.id}')
        run_df = pd.DataFrame(list(run.scan_history()))

        # Sends run data to Hugging Face Hub
        export_df_to_hf(run_df, hf_datasets_path, version, run)

        # Updates metadata info
        metadata_info_df.loc[run_id_metadata_row, 'completed'] = True
        metadata_info_df.loc[run_id_metadata_row, 'downloaded'] = True
        metadata_info_df.loc[run_id_metadata_row, 'last_checkpoint'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_info_df.to_json(f"hf://datasets/{hf_datasets_path}/{version}/metadata.json")
        return
    else:
        raise Exception(f'Run {run.id} has an unexpected state: {run.state}')


def collect_wandb_data(
    metadata_info_df: pd.DataFrame,
    wandb_project: str,
    version: str,
    hf_datasets_path: str
):
    api = wandb.Api()
    wandb.login(anonymous="allow")

    runs = api.runs(wandb_project, filters={"tags": {"$in": [ version ] }})

    runs_pbar = tqdm.tqdm(sorted(runs), desc="Loading history from wandb", total=len(runs), unit="run")

    for run in runs_pbar:
        try:
            run_id_metadata_row = metadata_info_df.loc[metadata_info_df['run_id'] == run.id]
            run_id_already_captured = len(run_id_metadata_row) != 0

            if run_id_already_captured:
                consume_wandb_run(
                    run=run,
                    wandb_project=wandb_project,
                    wandb_api=api,
                    hf_datasets_path=hf_datasets_path,
                    version=version,
                    metadata_info_df=metadata_info_df
                )
            else:
                bt.logging.info(f'New run with id {run.id} identified, capturing it...')
                # Creates new metadata row
                run_id_metadata_row = pd.DataFrame(
                    {
                        "run_id": run.id,
                        "completed": False,
                        "downloaded": False,
                        "last_checkpoint": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "hotkey": run.config["hotkey"],
                        "openvalidators_version": version
                    }
                )
                # Appends new metadata row to metadata dataframe
                metadata_info_df = metadata_info_df.append(run_id_metadata_row, ignore_index=True)

                # Sends metadata info to Hugging Face Hub
                metadata_info_df.to_json(f"hf://datasets/{hf_datasets_path}/{version}/metadata.json")

                bt.logging.info(f'Run {run.id} captured successfully! Consuming run...')

                consume_wandb_run(
                    run=run,
                    wandb_project=wandb_project,
                    wandb_api=api,
                    hf_datasets_path=hf_datasets_path,
                    version=version,
                    metadata_info_df=metadata_info_df
                )
        except Exception as e:
            bt.logging.error(f'Error while consuming run {run.id}: {e}')
            logger.add("error.log", format="{time} {level} {message}", level="ERROR")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect data from wandb and save to huggingface datasets")
    parser.add_argument("--wandb_project", type=str, help="Wandb project to crawl", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--version", type=str, help="Tag version of openvalidators to be crawled", default=DEFAULT_VERSION)
    parser.add_argument("--dataset_output_dir", type=str, help="Hugging Face dataset output directory", default=DEFAULT_HF_DATASET_OUTPUT_DIR)
    parser.add_argument("--hf_token", type=str, help="Hugging Face token", default=HF_TOKEN)
    args = parser.parse_args()

    # Login to hugging face
    login(args.hf_token)

    # Load metadata info
    metadata_info = load_metadata_info(args.version, args.dataset_output_dir)

    # Collecting wandb data
    collect_wandb_data(
        metadata_info_df=metadata_info,
        wandb_project=args.wandb_project,
        version=args.version,
        hf_datasets_path=args.dataset_output_dir
    )

    bt.logging.info(f"Runs from version {args.version} collected successfully!")


