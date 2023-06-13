# Openvalidators wandb data collector

This folder contain a set of scripts to automate the data collection of the openvalidators project in wandb.

## Requirements
Ensure that the [requirements](../requirements.txt) of the project are properly installed on your machine.

```bash
pip install -e ../setup.py # Install openvalidators from root folder
```

The current implementation requires you to be logged in wandb. You can do so by running the following command:
```bash
wandb login
```

You will also need to install `make` to profit from the predefined targets in the [Makefile](Makefile). 
To install `make` on Ubuntu, run the following command:
```bash
sudo apt install make
```

If you are on Mac, you can install it by running:
```bash
brew install make
```


**Note:** All the data collection scripts are designed to be run from the `scripts` directory, 
so ensure that you are in the right place before starting:
```bash
 cd scripts # From root folder
```

## Usage
This repository provides a convenient way to collect data from the
[WandB Openvalidators platform](https://wandb.ai/opentensor-dev/openvalidators) using the [data_collector.py](data_collector.py)
tool. The repository includes a [Makefile](Makefile) that simplifies the execution of commands for data extraction by providing 
predefined targets.

The [data_collector.py](data_collector.py) script is designed to extract data from WandB runs based on a given set of
parameters. It supports various options to specify the type of data to collect and the configuration for exporting the
data. 

The repository's Makefile includes the following targets to facilitate data collection:


### `openvalidators_dataset`

```bash
make openvalidators_dataset
```

This command downloads all the runs from the latest version of the project and exports them to a CSV file named 
**openvalidators_dataset.csv**.
It utilizes the following options under the hood of the [Makefile](Makefile):

- `--download_all`: Downloads all the runs.
- `--export_path`: Specifies the path and filename for the exported CSV file.

---

### `run_id_dataset`
```bash
make run_id_dataset RUN_ID=<run_id>
```
This command downloads a specific run from the project and exports it to a CSV file named 
**$(RUN_ID)_openvalidators_dataset.csv**. It utilizes the following options:

- `--export_path`: Specifies the path and filename for the exported CSV file.
- `--wandb_run_id`: Specifies the ID of the run to download.

---

### `mining_dataset`
```bash
make mining_dataset
```
This command downloads all the runs from the latest version of the project with a mining dataset and exports them to a 
CSV file named **openvalidators_dataset.csv**. It utilizes the following options:

- `--download_all`: Downloads all the runs.
- `--export_path`: Specifies the path and filename for the exported CSV file.
- `--export_mining_dataset`: Enables the export of mining dataset.

---

### `scored_mining_dataset`
```bash
make scored_mining_dataset
```

This command downloads all the runs from the latest version of the project with a scored mining dataset and exports them
to a CSV file named **openvalidators_dataset.csv**. It utilizes the following options:

- `--download_all`: Downloads all the runs.
- `--export_path`: Specifies the path and filename for the exported CSV file.
- `--export_mining_with_scoring_dataset`: Enables the export of mining dataset with scoring.

---

### `openai_mining_dataset`
```bash
make openai_mining_dataset
```

This command downloads all the runs from the latest version of the project and exports them to jsonl file named
**openai_mining_dataset_openvalidators.jsonl** in the [openai fine-tuning format](https://platform.openai.com/docs/guides/fine-tuning/prepare-training-data).

Note: Feel completely free to adjust the [data_collector.py](data_collector.py) script and [Makefile](Makefile) as necessary to 
match your project configuration and requirements.

---

# Data collector parameters

The data_collector.py script accepts the following parameters:

- **--download_all**: This parameter is a flag that, when set, instructs the script to download all the runs
**from the latest version of openvalidators.**
By default, it is set to False. Example usage: `--download_all`.
- **--include_tags**: This parameter allows you to specify a list of wandb tags to filter the data collection.
Example usage: `--include_tags=0.1.0,nfsw_filter,your-hot-key`.
- **--wandb_run_id**: This parameter allows you to specify the WandB run ID to download. It is used when you want to 
extract data from a specific run. By default, it is set to None. Example usage: `--wandb_run_id=ugutvtox`.
  > **Note:** The run_id can be retrieved at the end of your URL, e.g. :
  >  - URL: https://wandb.ai/opentensor-dev/openvalidators/runs/ugutvtox
  >  - Run_id: **ugutvtox**.
- **--export_mining_dataset**: This parameter is a flag that, when set, enables the export of the mining dataset. 
It is used when you want to extract data specifically for mining purposes, in the following format:
    ```json
    {
        "base_prompt" : "best_followup",
        "answer_prompt" : "best_answer",    
        //...
    }
    ```
    By default, it is set to False.
    Example usage: `--export_mining_dataset`.
- **--export_mining_with_scoring_dataset**: This parameter is a flag that, when set, enables the export of the mining 
dataset with scores, in the following format:
    ```json
    {
        "base_prompt" : { "best_followup" : "score" },
        "answer_prompt" : { "best_answer" : "score" },
        //...
    }
    ```
    It is used when you want to extract data for mining with scoring. By default, it is set to False.
    Example usage: `--export_mining_with_scoring_dataset`.

- **--export_path**: This parameter allows you to specify the path where the exported dataset will be saved.
By default, it is set to **"validator_dataset.csv"**. Example usage: `--export_path=your_path.csv`.
- **--mining_dataset_output_format:** This parameter allows you to specify the output format of the mining dataset. 
  Defaults to `json`, currently supports `json` and `csv`. Example usage: `--mining_dataset_output_format=csv`.
- **--blacklist_path**: This parameter allows you to specify the path to a file containing blacklist phrases.
The script will exclude any data that contains these phrases. By default, it is set to [blacklist_phrases.txt](blacklist_phrases.txt).
Example usage: `--blacklist_path=blacklist_phrases.txt`.
- **--export_openai_dataset**: This parameter is a flag that, when set, enables the export of the mining dataset
in the [jsonl openai format for fine-tuning](https://platform.openai.com/docs/guides/fine-tuning):
  ```json lines
  {"prompt": "base_prompt", "completion": "best_followup"},
  {"prompt": "answer_prompt", "completion": "best_answer" }
  ...
  ```

Make sure to adjust the parameters accordingly when executing the [data_collector.py](data_collector.py) script for your
specific data collection needs.