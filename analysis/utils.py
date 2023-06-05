# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import tqdm
import wandb

import pandas as pd
from pandas.api.types import is_list_like


def get_runs(project, filters=None, return_paths=False):
    api = wandb.Api()
    wandb.login()

    runs = api.runs(project, filters=filters)
    if return_paths:
        return [os.path.join(run.entity, run.project, run.id) for run in runs]
    else:
        return runs


def download_data(run_path=None, timeout=600):
    api = wandb.Api(timeout=timeout)
    wandb.login()

    if isinstance(run_path, str):
        run_path = [run_path]

    frames = []
    total_events = 0
    pbar = tqdm.tqdm(sorted(run_path), desc="Loading history from wandb", total=len(run_path), unit="run")
    for path in pbar:
        run = api.run(path)

        frame = pd.DataFrame(list(run.scan_history()))
        frames.append(frame)
        total_events += len(frame)

        pbar.set_postfix({"total_events": total_events})

    df = pd.concat(frames)
    # Convert timestamp to datetime.
    df._timestamp = pd.to_datetime(df._timestamp, unit="s")
    df.sort_values("_timestamp", inplace=True)

    return df


def load_data(path, nrows):

    df = pd.read_csv(path, nrows=nrows)

    # detect list columns which as stored as strings
    list_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].str.startswith("[").all()]

    # convert string representation of list to list
    df[list_cols] = df[list_cols].applymap(eval)

    return df


def explode_data(df, list_cols=None, list_len=None):

    if list_cols is None:
        list_cols = [c for c in df.columns if df[c].apply(is_list_like).all()]
        print(f"Exploding {len(list_cols)}) list columns with {list_len} elements: {list_cols}")
    if list_len:
        list_cols = [c for c in list_cols if df[c].apply(len).unique()[0] == list_len]
        print(f"Exploding {len(list_cols)}) list columns with {list_len} elements: {list_cols}")

    return df.explode(column=list_cols)


def get_list_col_lengths(df):
    list_col_lengths = {c: sorted(df[c].apply(len).unique()) for c in df.columns if df[c].apply(is_list_like).all()}
    varying_lengths = {c: v for c, v in list_col_lengths.items() if len(v) > 1}

    if len(varying_lengths) > 0:
        print(f"The following columns have varying lengths: {varying_lengths}")

    return {c: v[0] for c, v in list_col_lengths.items()}


def get_completions(series, ntop=10, completions=None, completion_regex=None, return_counts=False):

    if completions is None:
        completion_counts = series.value_counts()
        if completion_regex is not None:
            completions = completion_counts[completion_counts.index.str.contains(completion_regex)].index[:ntop]
            print(f"Using {len(completions)} completions which match {completion_regex!r}: \n{completions}")
        else:
            completions = completion_counts.index[:ntop]
            print(f"Using top {len(completions)} completions: \n{completions}")
    else:
        print(f"Using {len(completions)} completions: \n{completions}")
        completion_counts = series.loc[series.isin(completions)].value_counts()

    if return_counts:
        return completion_counts
    else:
        return completions
