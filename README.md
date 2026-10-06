# Grammar Scoring Engine for Spoken English

Predicts a continuous MOS grammar score (0–5) for a 45–60 second spoken response.
Built for the SHL Hiring Assessment 2026 (Kaggle).

The full report, code and results are in [`grammar_scoring_engine_final.ipynb`](grammar_scoring_engine_final.ipynb).

## Approach

Each clip is described by several independent views, a simple regressor is trained on each, and the predictions are stacked.

| View | How it is built |
|---|---|
| Listening | Whisper encoder (4 layers) and decoder state; WavLM-large (6 layers) |
| Reading | RoBERTa, DeBERTa and ELECTRA embeddings of a Whisper transcript |
| Examining | Hidden states of two small LLMs (Qwen2.5-3B, Phi-3.5-mini) prompted with the scoring rubric |
| Measuring | 27 interpretable grammar and fluency features, plus recogniser-agreement, CEFR and syntax features |

```
wav ─► noise guard ─────────────────────────────────────────► unintelligible clip → 0
    ─► audio, transcript, LLM and feature views ─► 54 base regressors (ridge, SVR, GBM)
    ─► 4 non-negative stacks ─► blend ─► pool a speaker's own clips ─► score
```

Findings from the data that shaped the design:

1. The 37 training clips labelled 0 are noise-corrupted recordings, not poor grammar. They are handled by a two-threshold rule and kept out of the regressors.
2. Speakers recur within the training set but no test speaker appears in it, so validation is grouped by speaker.
3. The test set uses different prompts (69% are 45-second tasks against 24% in training), so a second protocol holds out whole prompts.
4. Clips from the same speaker differ by about half a point, which bounds the achievable error.

## Results

| Evaluation | RMSE | Pearson r |
|---|---|---|
| Training set, in-sample (all 769 clips) | 0.250 | 0.980 |
| Speaker-grouped cross-validation (732 clean clips) | 0.510 | 0.865 |
| Held-out-prompt cross-validation | 0.528 | 0.855 |
| Baseline: predict the mean | 1.014 | – |

Public leaderboard score of `submission.csv`: **0.3211** (lower is better).

The in-sample figure shows fit, not generalisation. The cross-validated figures are the fair estimate for new speakers.

### What is validated and what is not

| Step | Status |
|---|---|
| Views, base regressors, stacking | Validated by speaker-grouped and held-out-prompt cross-validation |
| Pooling a speaker's own test clips | Direction validated; the strength used (0.65) is above the cross-validated optimum |
| Blend weights between the four stacks | Chosen using public-leaderboard feedback |
| Similar-voice adjustment | Not supported by training data; kept because it lowered the leaderboard score. Without it the same blend scores 0.3275 |

A five-model version with no leaderboard-derived step reaches RMSE 0.519 in the same cross-validation and scored 0.3355 on the leaderboard.

## Repository contents

| File | Description |
|---|---|
| `grammar_scoring_engine_final.ipynb` | Executed notebook: report, documented code, charts, evaluation |
| `submission.csv` | Predictions for the 216 test clips |
| `requirements.txt` | Python packages and versions used |

The competition dataset and the cached features are not included.

## Running the notebook

1. Install the packages in `requirements.txt` (PyTorch with CUDA support), then `python -m spacy download en_core_web_sm`.
2. Place the dataset so that `data/Dataset_Final/train.csv`, `test.csv` and the `train/` and `test/` audio folders sit next to the notebook.
3. Run all cells. Expensive steps are cached to `cache_final/`.

A from-scratch run takes roughly four hours on a 4 GB GPU. The features come from half-precision and 4-bit networks, so a fresh run reproduces the predictions closely but not digit for digit.
