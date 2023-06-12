import pyarrow as pa

DATASET_SCHEMA = pa.schema([
    ('_runtime', pa.float64()),
    ('_step', pa.int64()),
    ('_timestamp', pa.float64()),
    ('answer_completions', pa.list_(pa.string())),
    ('answer_prompt', pa.string()),
    ('answer_rewards', pa.list_(pa.float64())),
    ('answer_times', pa.list_(pa.float64())),
    ('answer_uids', pa.list_(pa.int32())),
    ('base_prompt', pa.string()),
    ('best_answer', pa.string()),
    ('best_followup', pa.string()),
    ('block', pa.float64()),
    ('followup_completions', pa.list_(pa.string())),
    ('followup_rewards', pa.list_(pa.float64())),
    ('followup_times', pa.list_(pa.float64())),
    ('followup_uids', pa.list_(pa.int64())),
    ('gating_loss', pa.float64()),
    ('gating_scorings', pa.list_(pa.float64())),
    ('moving_averaged_scores', pa.list_(pa.float64())),
    ('set_weights', pa.list_(pa.list_(pa.float64()))),
    ('step_length', pa.float64())
])