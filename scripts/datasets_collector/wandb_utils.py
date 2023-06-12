from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from schemas import DATASET_SCHEMA
from metadata import  get_wandb_metadata_info
from substrateinterface.base import is_valid_ss58_address

import bittensor as bt
import pandas as pd
import pyarrow as pa
import tqdm
import wandb
from loguru import logger


@dataclass
class ProblematicRun:
    run_id: str
    error: str
    run: "wandb_sdk.wandb_run.Run"

@dataclass
class CollectionOutputResult:
    problematic_runs: List[ProblematicRun]
    new_downloaded_run_ids: int
    skipped_run_ids: int

WANDB_DF_SCHEMA =['answer_rewards', 'moving_averaged_scores', '_step',
'gating_scorings', 'answer_times', 'followup_uids',
'answer_completions', 'followup_completions', 'followup_rewards',
'answer_uids', 'followup_times', 'step_length', '_runtime',
'base_prompt', 'best_answer', 'answer_prompt', 'gating_loss',
'best_followup', '_timestamp', 'block', 'set_weights']


def get_hotkey_from_tags(tags: List[str]) -> Optional[str]:
    for tag in tags:
        if is_valid_ss58_address(tag):
            return tag

    return None


def get_non_processed_runs(
    wandb_api: wandb.Api,
    wandb_project: str,
    version: str,
    metadata_info_df: pd.DataFrame
):
    wandb_runs = wandb_api.runs(wandb_project, filters={"tags": {"$in": [version]}})
    wandb_run_ids = [run.id for run in wandb_runs]

    all_processed_run_ids = []
    for _, metadata_run_id in metadata_info_df.iterrows():
        run_id_already_processed = (metadata_run_id['downloaded'] or metadata_run_id['problematic'])

        if metadata_run_id['run_id'] in wandb_run_ids and run_id_already_processed:
            all_processed_run_ids.append(metadata_run_id['run_id'])

    non_processed_runs = list(filter(lambda run: run.id not in all_processed_run_ids, wandb_runs))

    return non_processed_runs


def export_df_to_hf(
    run_df: pd.DataFrame,
    hf_datasets_path: str,
    version: str,
    run: "wandb_sdk.wandb_run.Run"
):
    expected_columns = set(WANDB_DF_SCHEMA)
    df_columns = set(run_df.columns.tolist())

    columns_mismatch = not set(expected_columns).issubset(df_columns)

    if columns_mismatch:
        bt.logging.error(f'Wandb Run dataset columns do not match the expected schema')
        raise Exception(f'run_df columns do not match the expected schema. \n \
                         Expected columns: {WANDB_DF_SCHEMA} \n \
                         Actual columns: {run_df.columns.tolist()}')

    run_id_file_output_path = f"hf://datasets/{hf_datasets_path}/{version}/raw_data/{run.id}.parquet"
    bt.logging.info(f'Exporting run data to {run_id_file_output_path}...')
    # TODO: add complementary metadata to the run_df

    # Needs to convert to pyarrow table with schema and reconvert to pandas dataframe to export to Hugging Face Hub
    pa.Table.from_pandas(run_df, schema=DATASET_SCHEMA)\
        .to_pandas()\
        .to_parquet(run_id_file_output_path)

    bt.logging.info(f'Run data exported successfully!')


def handle_problematic_run_ids(
    problematic_runs: List[ProblematicRun],
    metadata_info_df: pd.DataFrame,
    hf_datasets_path: str,
    version: str
):
    for problematic_run in problematic_runs:
        try:
            run_id_metadata_row = metadata_info_df.loc[metadata_info_df['run_id'] == problematic_run.run_id]
            run_id_already_captured = len(run_id_metadata_row) != 0

            if run_id_already_captured:
                metadata_info_df.loc[run_id_metadata_row.index, 'problematic'] = True
                metadata_info_df.loc[run_id_metadata_row.index, 'problematic_reason'] = problematic_run.error

                metadata_info_df.to_csv(f"hf://datasets/{hf_datasets_path}/{version}/metadata.csv", index=True, index_label="index")
            else:
                run_id_metadata_row = pd.DataFrame(
                    {
                        "run_id": problematic_run.run_id,
                        "completed": False,
                        "downloaded": False,
                        "last_checkpoint": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "hotkey": get_hotkey_from_tags(problematic_run.run.tags),
                        "openvalidators_version": version,
                        'problematic': True,
                        'problematic_reason': problematic_run.error
                    },
                    index=[len(metadata_info_df)]
                )

                run_id_metadata_row = pd.merge(run_id_metadata_row, get_wandb_metadata_info(problematic_run.run), left_index=True, right_index=True)

                # Concats new metadata row to metadata dataframe
                metadata_info_df = pd.concat([metadata_info_df, run_id_metadata_row])
                metadata_info_df.to_csv(f"hf://datasets/{hf_datasets_path}/{version}/metadata.csv", index=True, index_label="index")
                bt.logging.error('Problematic run_id updated successfully')
        except Exception as e:
            bt.logging.error(f'Error while handling problematic run {problematic_run.run_id}: {e}')
            logger.add("error.log", format="{time} {level} {message}", level="ERROR")
            continue


def consume_wandb_run(
    run: "wandb_sdk.wandb_run.Run",
    wandb_project: str,
    wandb_api: wandb.Api,
    hf_datasets_path: str,
    version: str,
    metadata_info_df: pd.DataFrame,
    collection_output_result: CollectionOutputResult
):
    run_id_metadata_row = metadata_info_df.loc[metadata_info_df['run_id'] == run.id]

    # Converts dataframe row to Metadata object
    run_id_metadata = run_id_metadata_row.iloc[0].to_dict()
    completed_states = ["finished", "failed", "crashed", "killed"]

    if run.state == "running":
        # Checks if run_id is still running. If so, skips run_id and updates last checkpoint
        bt.logging.info(f'Run {run.id} is still in progress, skipping and updating checkpoint...')

        # Updates last checkpoint of run_id
        metadata_info_df.loc[run_id_metadata_row.index, 'last_checkpoint'] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")

        metadata_info_df.to_csv(f"hf://datasets/{hf_datasets_path}/{version}/metadata.csv", index=True, index_label="index")
        collection_output_result.skipped_run_ids += 1
        return
    elif run.state in completed_states and not run_id_metadata["downloaded"]:
        # Checks if run_id is already completed but not downloaded. If so, downloads run_id and updates metadata
        bt.logging.info(f'Run {run.id} already completed with state {run.state}, downloading it...')

        run = wandb_api.run(f'{wandb_project}/{run.id}')
        run_df = pd.DataFrame(list(run.scan_history()))

        # Sends run data to Hugging Face Hub
        export_df_to_hf(run_df, hf_datasets_path, version, run)

        bt.logging.info(f'Updating metadata info...')
        # Updates metadata info
        metadata_info_df.loc[run_id_metadata_row.index, 'completed'] = True
        metadata_info_df.loc[run_id_metadata_row.index, 'downloaded'] = True
        metadata_info_df.loc[run_id_metadata_row.index, 'last_checkpoint'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_info_df.to_csv(f"hf://datasets/{hf_datasets_path}/{version}/metadata.csv", index=True, index_label="index")
        bt.logging.info(f'Metadata info updated successfully!')
        collection_output_result.new_downloaded_run_ids += 1
        return
    else:
        raise Exception(f'Run {run.id} has an unexpected state: {run.state}')


def collect_wandb_data(
    metadata_info_df: pd.DataFrame,
    wandb_project: str,
    version: str,
    hf_datasets_path: str
) -> CollectionOutputResult:
    api = wandb.Api()
    wandb.login(anonymous="allow")

    non_processed_runs = get_non_processed_runs(api, wandb_project, version, metadata_info_df)
    runs_pbar = tqdm.tqdm(non_processed_runs, desc="Loading unprocessed run_ids from wandb", total=len(non_processed_runs), unit="run")

    output_result = CollectionOutputResult(problematic_runs=[], skipped_run_ids=0, new_downloaded_run_ids=0)

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
                    metadata_info_df=metadata_info_df,
                    collection_output_result=output_result
                )
            else:
                bt.logging.info(f'New run with id {run.id} identified, capturing it...')

                # Creates new metadata row
                run_id_metadata = {
                    "run_id": run.id,
                    "completed": False,
                    "downloaded": False,
                    "last_checkpoint": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "hotkey": get_hotkey_from_tags(run.tags),
                    "openvalidators_version": version,
                    'problematic': False,
                    'problematic_reason': None
                }

                wandb_metadata = get_wandb_metadata_info(run)
                row_metadata = {**run_id_metadata, **wandb_metadata}

                run_id_metadata_row = pd.DataFrame.from_dict(row_metadata, orient='index').T

                # Concats new metadata row to metadata dataframe
                metadata_info_df = pd.concat([metadata_info_df, run_id_metadata_row])
                metadata_info_df.reset_index(inplace=True, drop=True)

                # Sends metadata info to Hugging Face Hub
                metadata_info_df.to_csv(f"hf://datasets/{hf_datasets_path}/{version}/metadata.csv", index=True, index_label="index")

                bt.logging.info(f'Run {run.id} captured successfully! Consuming run...')

                consume_wandb_run(
                    run=run,
                    wandb_project=wandb_project,
                    wandb_api=api,
                    hf_datasets_path=hf_datasets_path,
                    version=version,
                    metadata_info_df=metadata_info_df,
                    collection_output_result=output_result
                )
        except Exception as e:
            bt.logging.error(f'Error while consuming run {run.id}: {e}')
            problematic_run = ProblematicRun(run_id=run.id, run=run, error=str(e))
            output_result.problematic_runs.append(problematic_run)
            continue

    handle_problematic_run_ids(
        output_result.problematic_runs,
        metadata_info_df,
        hf_datasets_path,
        version)

    return output_result