import bittensor as bt
import pandas as pd
from typing import Optional
from dataclasses import dataclass, asdict
from hf_utils import check_file_exists

METADATA_COLUMNS = ["run_id", "completed", "downloaded", "last_checkpoint", "hotkey", "openvalidators_version",
                    "problematic", "problematic_reason", "wandb_json_config", "logged_rows", "wandb_run_name",
                    "wandb_user_info", "wandb_tags", "wandb_createdAt","wandb_heartbeatAt", "wandb_state", ]

@dataclass
class WandbMetadata:
    wandb_json_config: str
    wandb_run_name: str
    wandb_user_info: str
    wandb_tags: str
    wandb_createdAt: str
    wandb_heartbeatAt: str
    wandb_state: str


@dataclass
class MetadataInfo(WandbMetadata):
    run_id: str
    completed: bool
    downloaded: bool
    last_checkpoint: str
    hotkey: str
    openvalidators_version: str
    problematic: bool
    problematic_reason: Optional[str]
    logged_rows: int


def load_metadata_info(hf_datasets_path: str, version: str) -> pd.DataFrame:
    """Loads metadata info from Hugging Face Hub
    Args:
        hf_datasets_path (str): Hugging Face dataset output directory
        version (str): Version of the dataset to collect

    Returns:
        pd.DataFrame: Metadata info
    """
    metadata_path = f"datasets/{hf_datasets_path}/{version}/metadata.csv"
    repo_metadata_exists = check_file_exists(metadata_path)

    if repo_metadata_exists:
        bt.logging.info(f'Metadata file located at {metadata_path}, loading metadata file...')
        # Reads CSV file directly from Hugging Face Hub
        metadata_info_df = pd.read_csv(f"hf://{metadata_path}")
        bt.logging.info(f'Metadata file loaded successfully!')

        return metadata_info_df
    else:
        bt.logging.info(f'No metadata file located at {metadata_path}, new metadata file will be created...')
        columns = METADATA_COLUMNS
        return pd.DataFrame(columns=columns)


def get_wandb_metadata_info(wandb_run: "wandb_sdk.wandb_run.Run") -> WandbMetadata:
    """Gets wandb metadata info from wandb run
    Args:
        wandb_run (wandb_sdk.wandb_run.Run): Wandb run
    """
    wandb_metadata = WandbMetadata(
        wandb_json_config=wandb_run.json_config,
        wandb_run_name=wandb_run.name,
        wandb_user_info=wandb_run.user.username,
        wandb_tags=wandb_run.tags,
        wandb_createdAt=wandb_run.createdAt,
        wandb_heartbeatAt=wandb_run.heartbeatAt,
        wandb_state=wandb_run.state
    )

    return wandb_metadata


def append_metadata_info(metadata_info_df: pd.DataFrame, new_metadata_info: MetadataInfo) -> pd.DataFrame:
    """Appends new metadata info to metadata dataframe
    Args:
        metadata_info_df (pd.DataFrame): Metadata info dataframe
        new_metadata_info (MetadataInfo): New metadata info
    Returns:
        pd.DataFrame: Metadata dataframe with new metadata info
    """
    run_id_metadata_row = pd.DataFrame.from_dict(asdict(new_metadata_info), orient='index').T

    # Concat new metadata row to metadata dataframe
    new_metadata_info_df = pd.concat([metadata_info_df, run_id_metadata_row])
    new_metadata_info_df.reset_index(inplace=True, drop=True)

    return new_metadata_info_df
