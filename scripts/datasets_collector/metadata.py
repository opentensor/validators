import bittensor as bt
import pandas as pd
from huggingface_hub import HfFileSystem

METADATA_COLUMNS = ["run_id", "completed", "downloaded",
"last_checkpoint", "hotkey", "openvalidators_version",
'problematic', 'problematic_reason', "wandb_json_config",
"wandb_run_name", "wandb_user_info", "wandb_tags", "wandb_createdAt"]

def load_metadata_info(hf_datasets_path: str, version: str) -> pd.DataFrame:
    metadata_path = f"datasets/{hf_datasets_path}/{version}/metadata.csv"

    hf_file_system = HfFileSystem()
    repo_metadata = hf_file_system.glob(metadata_path)

    repo_metadata_exists = len(repo_metadata) != 0

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


def get_wandb_metadata_info(wandb_run: "wandb_sdk.wandb_run.Run") -> dict:
    wandb_metadata = {
        "wandb_json_config": wandb_run.json_config,
        "wandb_run_name": wandb_run.name,
        "wandb_user_info": wandb_run.user.username,
        "wandb_tags": wandb_run.tags,
        "wandb_createdAt": wandb_run.createdAt,
    }

    return wandb_metadata