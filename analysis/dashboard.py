import streamlit as st
import pandas as pd
import os
import re
import time
from analysis.utils import get_runs, download_data, get_list_col_lengths, explode_data
import analysis.plotting as plotting


# dendrite time versus completion length
# prompt-based completion score stats


DEFAULT_PROJECT = "openvalidators"
DEFAULT_FILTERS = {"tags": {"$in": ["1.0.0", "1.0.1", "1.0.2", "1.0.3", "1.0.4"]}}

@st.cache_data
def load_runs(project=DEFAULT_PROJECT, filters=DEFAULT_FILTERS, min_steps=10):
    runs = []
    msg = st.empty()
    for run in get_runs(project, filters):
        step = run.summary.get('_step',0)
        if step < min_steps:
            msg.warning(f'Skipped run `{run.name}` because it contains {step} events (<{min_steps})')
            continue

        duration = run.summary.get('_runtime')
        end_time = run.summary.get('_timestamp')
        # extract values for selected tags
        rules = {'hotkey': re.compile('^[0-9a-z]{48}$',re.IGNORECASE), 'version': re.compile('^\\d\.\\d+\.\\d+$'), 'spec_version': re.compile('\\d{4}$')}
        # tags = {k: tag for k,tag in zip(('hotkey','version','spec_version'),run.tags)}
        tags = {k: tag for k, rule in rules.items() for tag in run.tags if rule.match(tag)}
        # include bool flag for remaining tags
        tags.update({k: k in run.tags for k in ('mock','custom_gating_model','nsfw_filter','outsource_scoring','disable_set_weights')})

        runs.append({
            'state': run.state,
            'num_steps': step,
            'entity': run.entity,
            'id': run.id,
            'name': run.name,
            'project': run.project,
            'url': run.url,
            'path': os.path.join(run.entity, run.project, run.id),
            'start_time': pd.to_datetime(end_time-duration, unit="s"),
            'end_time': pd.to_datetime(end_time, unit="s"),
            'duration': pd.to_datetime(duration, unit="s"),
            # 'tags': run.tags,
            **tags
        })
    msg.empty()
    return pd.DataFrame(runs).astype({'state': 'category', 'hotkey': 'category', 'version': 'category', 'spec_version': 'category'})


@st.cache_data
def load_data(selected_runs, load=True, save=False):

    frames = []
    n_events = 0
    progress = st.progress(0, 'Loading data')
    for i, idx in enumerate(selected_runs.index):
        run = selected_runs.loc[idx]
        prog_msg = f'Loading data {i/len(selected_runs)*100:.0f}% ({i}/{len(selected_runs)} runs, {n_events} events)'

        file_path = f'data/history-{run.id}.csv'

        if load and os.path.exists(file_path):
            progress.progress(i/len(selected_runs),f'{prog_msg}... reading {file_path}')
            df = pd.read_csv(file_path)
            # filter out events with missing step length
            df = df.loc[df.step_length.notna()]

            # detect list columns which as stored as strings
            list_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].str.startswith("[").all()]
            # convert string representation of list to list
            df[list_cols] = df[list_cols].applymap(eval, na_action='ignore')

        else:
            try:
                # Download the history from wandb
                progress.progress(i/len(selected_runs),f'{prog_msg}... downloading `{run.path}`')
                df = download_data(run.path)
                df.assign(**run.to_dict())
                if not os.path.exists('data/'):
                    os.makedirs(file_path)

                if save and run.state != 'running':
                    df.to_csv(file_path, index=False)
                    # st.info(f'Saved history to {file_path}')
            except Exception as e:
                st.error(f'Failed to download history for `{run.path}`')
                st.exception(e)
                continue

        frames.append(df)
        n_events += df.shape[0]

    progress.empty()
    # Remove rows which contain chain weights as it messes up schema
    return pd.concat(frames)

@st.cache_data
def get_exploded_data(df):
    list_cols = get_list_col_lengths(df)
    return explode_data(df, list(list_cols))

@st.cache_data
def get_completions(df_long, col):
    return df_long[col].value_counts()

@st.cache_data
def plot_uid_diversty(df, remove_unsuccessful=True):
    return plotting.plot_uid_diversty(df, remove_unsuccessful=remove_unsuccessful)

@st.cache_data
def plot_leaderboard(df, ntop, group_on, agg_col, agg, alias=False):
    return plotting.plot_leaderboard(df, ntop=ntop, group_on=group_on, agg_col=agg_col, agg=agg, alias=alias)

@st.cache_data
def plot_completion_rewards(df, completion_col, reward_col, uid_col, ntop, completions=None, completion_regex=None):
    return plotting.plot_completion_rewards(df, msg_col=completion_col, reward_col=reward_col, uid_col=uid_col, ntop=ntop, completions=completions, completion_regex=completion_regex)

@st.cache_data
def uid_metrics(df_long, src, uid=None):

    uid_col = f'{src}_uids'
    completion_col = f'{src}_completions'
    nsfw_col = f'{src}_nsfw_scores'
    reward_col = f'{src}_rewards'

    if uid is not None:
        df_long = df_long.loc[df_long[uid_col] == uid]

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Success %",
        value=f'{df_long.loc[df_long[completion_col].str.len() > 0].shape[0]/df_long.shape[0] * 100:.1f}'
    )
    col2.metric(
        label="Diversity %",
        value=f'{df_long[completion_col].nunique()/df_long.shape[0] * 100:.1f}'
    )
    col3.metric(
        label="Toxicity %",
        value=f'{df_long[nsfw_col].mean() * 100:.1f}' if nsfw_col in df_long.columns else 'N/A'
    )

st.title('Validator :red[Analysis] Dashboard :eyes:')
# add vertical space
st.markdown('#')
st.markdown('#')


with st.sidebar:
    st.sidebar.header('Pages')

with st.spinner(text=f'Checking wandb...'):
    df_runs = load_runs()
# get rows where start time is older than 24h ago
df_runs_old = df_runs.loc[df_runs.start_time < pd.to_datetime(time.time()-24*60*60, unit='s')]

col1, col2, col3 = st.columns(3)

col1.metric('Runs', df_runs.shape[0], delta=f'{df_runs.shape[0]-df_runs_old.shape[0]} (24h)')
col2.metric('Hotkeys', df_runs.hotkey.nunique(), delta=f'{df_runs.hotkey.nunique()-df_runs_old.hotkey.nunique()} (24h)')
col3.metric('Events', df_runs.num_steps.sum(), delta=f'{df_runs.num_steps.sum()-df_runs_old.num_steps.sum()} (24h)')

# https://wandb.ai/opentensor-dev/openvalidators/runs/kt9bzxii/overview?workspace=
# all_run_paths = ['opentensor-dev/openvalidators/kt9bzxii'] # pedro long run

run_ids = df_runs.id
default_selected_runs = ['kt9bzxii']
selected_runs = default_selected_runs

# add vertical space
st.markdown('#')
st.markdown('#')


tab1, tab2, tab3, tab4 = st.tabs(["Wandb Runs", "UID Health", "Completions", "Prompt-based scoring"])

# src = st.radio('Choose data source:', ['followup', 'answer'], horizontal=True, key='src')
# list_list_cols = get_list_col_lengths(df_long)
# df_long_long = explode_data(df_long, list(list_list_cols))

with tab1:

    st.markdown('#')
    st.subheader(":violet[Wandb] Runs")

    # Load data
    df = load_data(df_runs.loc[run_ids.isin(selected_runs)], load=True, save=True)
    df_long = get_exploded_data(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Selected runs", value=len(selected_runs))
    col2.metric(label="Events", value=df.shape[0]) #
    col3.metric(label="UIDs", value=df_long.followup_uids.nunique())
    col4.metric(label="Unique completions", value=df_long.followup_uids.nunique())

    selected_runs = st.multiselect(f'Runs ({len(df_runs)})', run_ids, default=selected_runs)

    st.markdown('#')
    st.subheader("View :violet[Data]")

    show_col1, show_col2 = st.columns(2)
    show_runs = show_col1.checkbox('Show runs', value=True)
    show_events = show_col2.checkbox('Show events', value=False)
    if show_runs:
        st.markdown(f'Wandb info for **{len(selected_runs)} selected runs**:')
        st.dataframe(df_runs.loc[run_ids.isin(selected_runs)],
                    column_config={
                        "url": st.column_config.LinkColumn("URL"),
                    }
        )

    if show_events:
        st.markdown(f'Raw events for **{len(selected_runs)} selected runs**:')
        st.dataframe(df.head(50),
                    column_config={
                        "url": st.column_config.LinkColumn("URL"),
                    }
        )

default_src = 'followup'
with tab2:

    st.markdown('#')
    st.subheader("UID :violet[Health]")
    uid_src = default_src

    # uid = st.selectbox('UID:', sorted(df_long[uid_col].unique()), key='uid')

    uid_metrics(df_long, uid_src)
    uid_src = st.radio('Select one:', ['followup', 'answer'], horizontal=True, key='uid_src')
    uid_col = f'{uid_src}_uids'
    reward_col = f'{uid_src}_rewards'

    st.markdown('#')
    st.subheader("UID :violet[Leaderboard]")
    uid_ntop_default = 10

    uid_col1, uid_col2 = st.columns(2)
    uid_ntop = uid_col1.slider('Number of UIDs:', min_value=1, max_value=50, value=uid_ntop_default, key='uid_ntop')
    uid_agg = uid_col2.selectbox('Aggregation:', ('mean','min','max','size','nunique'), key='uid_agg')

    st.plotly_chart(
        plot_leaderboard(
            df,
            ntop=uid_ntop,
            group_on=uid_col,
            agg_col=reward_col,
            agg=uid_agg
        )
    )
    remove_unsuccessful = st.checkbox('Remove failed completions', value=True)
    st.plotly_chart(
        plot_uid_diversty(
            df,
            remove_unsuccessful=remove_unsuccessful
        )
    )


completion_ntop_default = 10
with tab3:

    st.markdown('#')
    st.subheader('Completion :violet[Leaderboard]')
    completion_src = default_src

    msg_col1, msg_col2 = st.columns(2)
    completion_src = msg_col1.radio('Select one:', ['followup', 'answer'], horizontal=True, key='completion_src')
    completion_ntop = msg_col2.slider('Top k:', min_value=1, max_value=50, value=completion_ntop_default, key='completion_ntop')

    completion_col = f'{completion_src}_completions'
    reward_col = f'{completion_src}_rewards'
    uid_col = f'{completion_src}_uids'

    completions = get_completions(df_long, completion_col)

    # completion_sel = st.radio('Select input method:', ['ntop', 'select','regex'], horizontal=True, key='completion_sel')
    # Get completions with highest average rewards
    st.plotly_chart(
        plot_leaderboard(
            df,
            ntop=completion_ntop,
            group_on=completion_col,
            agg_col=reward_col,
            agg='mean',
            alias=True
        )
    )
    st.markdown('#')
    st.subheader('Completion :violet[Rewards]')

    completion_select = st.multiselect('Completions:', completions.index, default=completions.index[:3].tolist())
    # completion_regex = st.text_input('Completion regex:', value='', key='completion_regex')

    st.plotly_chart(
        plot_completion_rewards(
            df,
            completion_col=completion_col,
            reward_col=reward_col,
            uid_col=uid_col,
            ntop=completion_ntop,
            completions=completion_select,
        )
    )

with tab4:
    st.subheader(':pink[Prompt-based scoring]')
    prompt_src = st.radio('Select one:', ['followup', 'answer'], key='prompt')


    # st.dataframe(df_long_long.filter(regex=prompt_src).head())

