import argparse
import tqdm
import json
import pandas as pd
from analysis.utils import get_runs, download_data


DEFAULT_PROJECT = 'openvalidators'

def collect_data(download_all: bool, export_path: str, wandb_run_id: str = None) -> pd.DataFrame:
    if download_all:
        runs = get_runs(DEFAULT_PROJECT, return_paths=True)
        df = download_data(runs)
        df.to_csv(export_path)
    else:
        if wandb_run_id is None:
            raise Exception("Please specify a wandb run id to download")
        else:
            df = download_data(f'{DEFAULT_PROJECT}/{wandb_run_id}')
            df.to_csv(export_path)

    print(f'Data collected successfully at: {export_path}')
    return df


def create_mining_dataset(df: pd.DataFrame, export_path: str, with_score=False):
    mining_export_dataset = f'mining_dataset.json'
    dict_dataset = {}

    for _, row in tqdm.tqdm(df.iterrows(), desc='Creating mining dataset', total=len(df), unit='run'):
        base_prompt = row['base_prompt']
        best_followup = row['best_followup']

        answer_prompt = row['answer_prompt']
        best_answer = row['best_answer']

        if with_score:
            pass
        else:
            dict_dataset[base_prompt] = best_followup
            dict_dataset[answer_prompt] = best_answer

    with open(mining_export_dataset, 'w') as json_file:
        json.dump(dict_dataset, json_file)

    print(f"Mining dataset exported successfully to {export_path}")


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the flags as parameters
    parser.add_argument("--download_all", action="store_true", help="Downloads all runs from project", default=False)
    parser.add_argument("--wandb_run_id", type=str, help="Specify the wandb run id to download", default=None)
    parser.add_argument("--export_mining_dataset", action="store_true", help="Exports the mining dataset", default=False)
    parser.add_argument("--export_mining_with_scoring_dataset", action="store_true", help="Exports mining dataset with scores", default=False)
    parser.add_argument("--export_path", type=str, help="Specify the path to export the dataset", default="validator_dataset.csv")

    args = parser.parse_args()

    download_all = args.download_all
    wandb_run_id = args.wandb_run_id
    export_mining_dataset = args.export_mining_dataset
    export_mining_with_scoring_dataset = args.export_mining_with_scoring_dataset
    export_path = args.export_path

    collected_data = collect_data(download_all, export_path, wandb_run_id)

    if export_mining_dataset or export_mining_with_scoring_dataset:
        create_mining_dataset(df=collected_data, export_path=export_path, with_score=export_mining_with_scoring_dataset)









