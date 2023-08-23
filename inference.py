import sys
import asyncio
import random
import pandas as pd

import streamlit as st
import plotly.express as px

from types import SimpleNamespace
from typing import List


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


from openvalidators.neuron import neuron
from openvalidators.forward import run_step

# create streamlit app in wide mode
# st.set_page_config(layout="wide")

st.title("OpenValidators Live Inference")
st.markdown("<br>", unsafe_allow_html=True)

st.info(
    "This is a UI for running inference on the OpenValidators network. It is intended to be used for testing and debugging reward models behavior."
)
# add verticle space
# st.markdown('<br>', unsafe_allow_html=True)
st.markdown("---")


# fixed for now
num_completions = 4
num_columns = 2
completions = [""] * num_completions

# Override the async_forward function so that the UI can be used to provide completions.
async def ui_async_forward(
    roles: List[str],
    messages: List[str],
    uids: List[int],
    timeout: float = 12,
    return_call=True,
):
    async def query():
        await asyncio.sleep(0.01)
        return [SimpleNamespace(completion=completion, elapsed_time=0, is_success=True) for completion in completions]

    return await query()


async def inference(validator, prompt):
    # Request a response to the test prompt
    event = await run_step(
        validator,
        prompt=prompt,
        name="test",
        k=num_completions,
        timeout=5,
    )
    print(f"Event: {event}")
    return pd.DataFrame(event)


@st.cache_resource()
def load_validator():
    # Validator will be created with CLI args
    sys.argv += ["--mock", "--neuron.axon_off", "--wandb.off", "--neuron.dont_save_events"]  # ,'--neuron.mock_dendrite_pool']
    return neuron()


validator = load_validator()
# override dendrite pool async forward
validator.dendrite_pool.async_forward = ui_async_forward


def get_prompt():
    # Obtain a unique context from the dataset.
    prompt = next(validator.dataset)["text"]
    print(f"Context: {prompt!r}")
    if prompt.count("."):
        random_cutoff = random.randint(15, 30)
        # Truncate context to a limited set of sentences.
        prompt = ".".join(prompt.split(".", maxsplit=random_cutoff)[:-1])

    print(f"Prompt: {prompt!r}")
    return prompt


with st.expander("Configure prompt and completions"):
    base_prompt = get_prompt()

    prompt_col1, prompt_col2 = st.columns([0.8, 0.2])

    with prompt_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        new_prompt_button = st.button("New prompt")
        if new_prompt_button:
            base_prompt = get_prompt()
            user_prompt = base_prompt
        reset_prompt_button = st.button("Reset prompt")
        if reset_prompt_button:
            user_prompt = base_prompt

    user_prompt = prompt_col1.text_area("Prompt", value=base_prompt, height=100, max_chars=1000, key="prompt")
    print(f"User prompt: {user_prompt!r}")

    # allow user to select number of completions to generate
    # num_completions = st.slider('Number of completions', min_value=1, max_value=10, value=3, step=1)
    # create a list of K editable text boxes for completions
    num_rows = int()
    for i in range(num_completions // num_columns):
        for j, c in enumerate(st.columns(num_columns)):
            print(f"Completion {i*num_columns+j}, i {i}, j {j}")
            index = i * num_columns + j
            completions[index] = c.text_area(
                f"Completion {index}", value="", height=100, max_chars=1000, key=f"completion-{index}"
            )

    st.markdown("---")

# TODO: Allow either single step or full forward
run_col1, run_col2, run_col3 = st.columns(3)
inference_type = run_col1.selectbox("Step type", ["Full", "Augment", "Followup", "Answer"])

run_col3.markdown("<br>", unsafe_allow_html=True)
run_inference_button = run_col3.button("Run inference", type="primary")

if run_inference_button:
    df = asyncio.run(inference(validator, user_prompt))
    df.reset_index(inplace=True)

    reward_cols = list(df.filter(regex="reward").columns)
    filter_cols = list(df.filter(regex="filter").columns)
    for col in reward_cols:
        df[col] = [random.uniform(0, 1) for _ in range(len(df))]
    for col in filter_cols:
        df[col] = [random.choice([1, 0]) for _ in range(len(df))]

    st.dataframe(df)

    fig = px.parallel_coordinates(
        df,
        color="rewards",
        dimensions=filter_cols + reward_cols,
        labels={"rewards": "Reward"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show reward plots"):
        chart_type = st.radio("Select plot type", ["polar", "bar"], index=1, horizontal=True)
        for col in df.filter(regex="reward|filter").columns:
            title = col.title().replace("_", " ")
            st.text(title)
            if chart_type == "polar":
                fig = px.line_polar(
                    df.astype({col: float}).round(3),
                    r=col,
                    theta="index",
                    color="completions",
                    labels={"completions": "Completion", "index": "", col: title},
                    line_close=True,
                    height=400,
                )
            else:
                fig = px.bar(
                    df.astype({col: float}).round(3),
                    x=col,
                    y="index",
                    text="completions",
                    labels={"completions": "Completion", "index": "", col: title},
                    orientation="h",
                    color=col,
                    height=400,
                )
            st.plotly_chart(fig, use_container_width=True)
