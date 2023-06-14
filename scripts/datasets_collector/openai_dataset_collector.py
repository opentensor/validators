import bittensor as bt
import pandas as pd
import tqdm
from datasets import disable_progress_bar
from huggingface_hub import login
from typing import List
from hf_utils import check_file_exists
from metadata import load_metadata_info

disable_progress_bar()


SCANNED_RUNS_COLUMNS = ["run_id", "unique_rows_contrib"]
SUPPORTED_VERSIONS = ['1.0.0', '1.0.1', '1.0.2', '1.0.3', '1.0.4']
DEFAULT_HF_SOURCE_DATASET = 'opentensor/openvalidators-test'
DEFAULT_HF_DATASET_OUTPUT_DIR = 'pedroferreira/test-release'

OPENAI_DATASET_PATH = 'openai/openvalidators-openai.jsonl'


def get_scanned_runs_df(hf_datasets_path: str) -> pd.DataFrame:
    scanned_runs_path = f"datasets/{hf_datasets_path}/openai/scanned_runs.csv"

    repo_metadata_exists = check_file_exists(scanned_runs_path)

    if repo_metadata_exists:
        bt.logging.info(f'Scanned runs file located at {scanned_runs_path}, loading file...')
        # Reads CSV file directly from Hugging Face Hub
        scanned_runs_df = pd.read_csv(f"hf://{scanned_runs_path}")
        bt.logging.info(f'Scanned runs loaded successfully!')

        return scanned_runs_df
    else:
        bt.logging.info(f'No scanned run file located at {scanned_runs_path}, new file will be created...')
        new_scanned_runs_df = pd.DataFrame(columns=SCANNED_RUNS_COLUMNS)
        new_scanned_runs_df.to_csv(f"hf://{scanned_runs_path}", index=False)

        return new_scanned_runs_df


def get_downloaded_runs(hf_data_source_path: str, version: str) -> List[str]:
    version_metadata_info = load_metadata_info(hf_datasets_path=hf_data_source_path, version=version)

    downloaded_runs = version_metadata_info[version_metadata_info['downloaded'] == True]
    downloaded_runs_ids = downloaded_runs['run_id'].tolist()

    return downloaded_runs_ids


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates(subset=df.columns)
    df = df.reset_index(drop=True)

    return df



def is_valid_pair(openapi_dataset_df: pd.DataFrame, prompt, completion) -> bool:
    prompt_invalid = prompt is None or prompt == ''
    completion_invalid = completion is None or completion == ''

    if prompt_invalid or completion_invalid:
        return False

    if len(openapi_dataset_df) == 0:
        return True
    else:
        existing_prompt = openapi_dataset_df['prompt'] == prompt
        existing_completion = openapi_dataset_df['completion'] == completion

        # Filter matching pairs that are equal to prompt and completion
        result = openapi_dataset_df[existing_prompt & existing_completion]

        if not result.empty:
            return False

    return True


def get_new_samples(
    openai_dataset_df: pd.DataFrame,
    hf_data_source_path: str,
    run_id: str,
    version: str,
    blacklist: List[str]
) -> pd.DataFrame:
    hf_run_id_path = f'hf://datasets/{hf_data_source_path}/{version}/raw_data/{run_id}.parquet'
    run_id_dataset = pd.read_parquet(hf_run_id_path)

    bt.logging.info(f'Run {run_id} loaded with {len(run_id_dataset)} records...')

    prompt_best_followup_df = clean_df(run_id_dataset[['base_prompt', 'best_followup']])
    answer_best_answer_df = clean_df(run_id_dataset[['answer_prompt', 'best_answer']])

    concatenated_df = pd.concat([prompt_best_followup_df, answer_best_answer_df], axis=1).fillna('')

    new_samples_dict = {
        "prompt": [],
        "completion": []
    }

    for _, row in concatenated_df.iterrows():
        base_prompt = row['base_prompt']
        best_followup = row['best_followup']

        answer_prompt = row['answer_prompt']
        best_answer = row['best_answer']

        if best_followup not in blacklist and is_valid_pair(openai_dataset_df, base_prompt, best_followup):
            new_samples_dict['prompt'].append(base_prompt)
            new_samples_dict['completion'].append(best_followup)

        if best_answer not in blacklist and is_valid_pair(openai_dataset_df, base_prompt, best_followup):
            new_samples_dict['prompt'].append(answer_prompt)
            new_samples_dict['completion'].append(best_answer)

    new_samples_df = pd.DataFrame.from_dict(new_samples_dict)
    return new_samples_df


def load_openai_dataset(hf_datasets_path: str, dataset_path: str) -> pd.DataFrame:
    openai_dataset_path = f"datasets/{hf_datasets_path}/{dataset_path}"

    files_exists = check_file_exists(openai_dataset_path)

    if files_exists:
        openai_dataset_df = pd.read_json(f'hf://datasets/{DEFAULT_HF_DATASET_OUTPUT_DIR}/{OPENAI_DATASET_PATH}', lines=True)
        return openai_dataset_df

    new_openai_dataset_df = pd.DataFrame.from_dict({
        "prompt": [],
        "completion": []
    })

    return new_openai_dataset_df

def append_scanned_runs_batch(scanned_runs_df: pd.DataFrame, run_id: str, sample_size: int) -> pd.DataFrame:
    new_scanned_runs_df = pd.DataFrame.from_dict({
        'run_id': run_id,
        'unique_rows_contrib': sample_size
    }, orient='index').T

    appended_scanned_runs_df = pd.concat([scanned_runs_df, new_scanned_runs_df])
    appended_scanned_runs_df.reset_index(inplace=True, drop=True)

    return appended_scanned_runs_df


def main(version):

    scanned_runs_df = get_scanned_runs_df(hf_datasets_path=DEFAULT_HF_DATASET_OUTPUT_DIR)
    scanned_ids = []# scanned_runs_df['run_id'].tolist()

    downloaded_runs_ids = get_downloaded_runs(hf_data_source_path=DEFAULT_HF_SOURCE_DATASET, version=version)

    new_runs_ids = list(set(downloaded_runs_ids) - set(scanned_ids))

    bt.logging.info(f'Number of new runs to be ingested: {len(new_runs_ids)}')

    openai_dataset_df = load_openai_dataset(hf_datasets_path=DEFAULT_HF_DATASET_OUTPUT_DIR, dataset_path=OPENAI_DATASET_PATH)

    try:
        for run_id in tqdm.tqdm(new_runs_ids, desc='Ingesting runs', total=len(new_runs_ids),unit='run'):
            bt.logging.info(f'Ingesting run {run_id}...')
            new_samples_df = get_new_samples(
                openai_dataset_df=openai_dataset_df,
                hf_data_source_path=DEFAULT_HF_SOURCE_DATASET,
                run_id=run_id,
                version=version,
                blacklist=[])

            # Append new samples to the openai dataset
            openai_dataset_df = pd.concat([openai_dataset_df, new_samples_df])

            bt.logging.info(f'Run {run_id} collected with {len(new_samples_df)} new samples...')
            bt.logging.info(f'OpenAI dataset size: {len(openai_dataset_df)}')

            # Append new run_id to the scanned runs
            scanned_runs_df = append_scanned_runs_batch(
                scanned_runs_df=scanned_runs_df,
                run_id=run_id,
                sample_size=len(new_samples_df))

        openai_dataset_df = openai_dataset_df.sort_values(by=['prompt']).drop_duplicates()
        openai_dataset_df.to_json(f'hf://datasets/{DEFAULT_HF_DATASET_OUTPUT_DIR}/{OPENAI_DATASET_PATH}', orient='records', lines=True)
        scanned_runs_df.to_csv(f"hf://datasets/{DEFAULT_HF_DATASET_OUTPUT_DIR}/openai/scanned_runs.csv", index=False)
    except Exception as e:
        bt.logging.error(f'Error while ingesting new runs: {e}')
        raise e




if __name__ == "__main__":
    login('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')

    for version in SUPPORTED_VERSIONS:
        bt.logging.info('Starting version ', version)
        main(version)
