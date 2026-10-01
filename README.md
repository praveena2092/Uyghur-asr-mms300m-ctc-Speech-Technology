# Uyghur ASR with MMS-300M (CTC)

Full fine-tuning of [`facebook/mms-300m`](https://huggingface.co/facebook/mms-300m), a multilingual wav2vec2-style speech model, with a character-level CTC head for **Uyghur automatic speech recognition**.

## Task

Transcribe Uyghur speech clips. Metric: **Character Error Rate (CER)**, i.e. Levenshtein distance divided by reference length (lower is better).

| | |
|---|---|
| Audio | 9,468 mono 16 kHz clips, ~23.95 h |
| Train / test clips | 7,574 / 1,894 |
| Transcript script | Latin-script romanization of Uyghur |
| Vocabulary | 36 tokens (characters + `\|` word delimiter + `[UNK]` + `[PAD]`) |

## Approach

1. **Text normalization** – Unicode NFC, whitespace collapsing (light, to avoid altering meaning).
2. **Character vocabulary** built from training transcripts; `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` combined into a `Wav2Vec2Processor`.
3. **Audio preprocessing** – resample to 16 kHz, normalize; clips longer than 20 s are dropped from training (322 clips) for memory safety.
4. **Full fine-tune** (all 315M parameters, including the feature encoder) with a freshly initialized CTC head. `mms-300m` has no language adapter, so full fine-tuning is the appropriate route.
5. **Model selection** – 95/5 train/validation split, checkpoint chosen by lowest validation CER, early stopping (patience 4).
6. **Inference** – batched greedy CTC decoding in fp16 over `test.csv`, written as `ID,transcription`.

### Hyperparameters

| Setting | Value |
|---|---|
| Base model | facebook/mms-300m |
| Epochs | 4 |
| Learning rate | 3e-5, 10% warmup |
| Batch size | 1 (no gradient accumulation) |
| Weight decay | 0.005 |
| Regularization | dropout 0.1, SpecAugment `mask_time_prob=0.05`, layerdrop 0.05 |
| Precision | fp16 |
| Seed | 42 |

## Results

| Split | CER |
|---|---|
| Validation (379 held-out clips) | **0.0488** |

Training took about 2 hours on a single GPU (≈7,300 s for 27,492 steps).

## Repository layout

```
notebooks/mms300m_uyghur_ctc_finetune.ipynb   # full pipeline: data -> train -> submission.csv
data/                                          # put dataset here (git-ignored)
requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
```

Open the notebook, set the paths in the `CFG` cell, and run all cells. It writes `submission.csv` with columns `ID,transcription`. A GPU is required for practical training times.

## Ideas to lower CER

- Beam search with a KenLM character/word language model via `pyctcdecode` (usually the largest gain over greedy decoding).
- Tune SpecAugment (`mask_time_prob`, `mask_feature_prob`) and learning rate/warmup.
- Larger batch size via gradient accumulation; train longer.
- Compare against Whisper fine-tuning or `mms-1b` variants.

## License

Code: MIT. The dataset is licensed CC BY-NC-SA 4.0 and is not included in this repository.
