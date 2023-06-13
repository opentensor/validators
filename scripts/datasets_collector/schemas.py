import pyarrow as pa

DATASET_SCHEMA = pa.schema([
    ('_runtime', pa.float64()), # Runtime of the event
    ('_step', pa.int64()), # Step of the event
    ('_timestamp', pa.float64()), # Timestamp of the event
    ('answer_completions', pa.list_(pa.string())), # Completions of the answer_prompt
    ('answer_prompt', pa.string()), # Prompt used to generate the answer
    ('answer_rewards', pa.list_(pa.float64())), # Rewards of the answer responses
    ('answer_times', pa.list_(pa.float64())), # Ellapsed time of answer responses
    ('answer_uids', pa.list_(pa.int32())), # UIDs of nodes that answered the answer_prompt
    ('base_prompt', pa.string()), # Bootstrap prompt
    ('best_answer', pa.string()), # Best answer response
    ('best_followup', pa.string()), # Best followup response
    ('block', pa.float64()), # Subtensor current block
    ('followup_completions', pa.list_(pa.string())), # Completions of the base_prompt
    ('followup_rewards', pa.list_(pa.float64())), # Rewards of the followup responses
    ('followup_times', pa.list_(pa.float64())), # Ellapsed time of followup responses
    ('followup_uids', pa.list_(pa.int64())), # UIDs of nodes that answered the base_prompt
    ('gating_loss', pa.float64()), # Gating model loss
    ('gating_scorings', pa.list_(pa.float64())), # Gating model scores
    ('moving_averaged_scores', pa.list_(pa.float64())), # Moving averaged scores at the time of the event
    ('set_weights', pa.list_(pa.list_(pa.float64()))),  # Processed weights of nodes by uid
    ('step_length', pa.float64()) # Time difference from beginning of forward call to event logging
])