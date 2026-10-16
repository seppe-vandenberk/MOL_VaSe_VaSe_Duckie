## CSV columns

The first three columns describe the saved training checkpoint. The next five
are individual evaluation metrics, followed by action counts and termination
counts. `model` and `run` identify each row. Training metadata comes from the
JSON stored inside each model ZIP, so renamed files such as `baseline.zip` work.

| Column | Meaning |
| --- | --- |
| `learning_rate` | Learning rate saved in the checkpoint's training configuration. |
| `speed` | Speed saved in the checkpoint's training configuration. ; it is not a measured velocity in metres per second. |
| `training_steps` | Actual training steps at this checkpoint (`num_timesteps`), rather than the requested training budget. For this project's warm starts, these are the steps added during the new training run. |
| `steps` | Environment steps executed during this evaluation run. |
| `reward_total` | Sum of all rewards returned by the environment during this evaluation run, including penalties. Uses the model's saved reward settings. |
| `reward_per_step` | `reward_total / max(steps, 1)`. |
| `both_lines_fraction` | Fraction of evaluation steps where `detection_state` was `both`: both lane boundaries formed a valid pair. Values range from 0 to 1. |
| `longest_clean_streak` | Largest consecutive count of steps without `terminated=True`. A failure ends the streak; a time-limit truncation does not. This does not require both boundaries to be visible on every step. |
| `action_count_right` | Number of full-right actions (action 0). |
| `action_count_slight_right` | Number of slight-right actions (action 1). |
| `action_count_straight` | Number of straight actions (action 2). |
| `action_count_slight_left` | Number of slight-left actions (action 3). |
| `action_count_left` | Number of full-left actions (action 4). The five action columns sum to `steps`, including initial steering perturbations. |
| `termination_reason_count` | Total failure terminations in this evaluation run: the sum of the reason-specific columns. Equals `failures`; excludes time-limit truncations. |
| `termination_reason_count::<reason>` | One numeric column per distinct failure reason seen anywhere in this batch. The exact reason text follows `::`, for example `termination_reason_count::lane not recovered for 40 steps`. Zero means this reason did not occur in that run. A missing reason is recorded as `unknown` by the evaluator. If the batch has no failures, no reason-specific columns are needed and the total remains zero. |
| `model` | Model ZIP filename, including the checkpoint suffix. |
| `run` | One-based evaluation run number within this model's report (normally 1–5). |
