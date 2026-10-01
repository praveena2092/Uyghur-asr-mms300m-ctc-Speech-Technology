# Data

Place the competition data here (git-ignored, not redistributed):

```
data/
├── train.csv     # ID, filepath, transcription   (7,574 clips)
├── test.csv      # ID, filepath                  (1,894 clips)
├── sample.csv    # ID, transcription             (submission format)
└── wavs/         # 9,468 mono 16 kHz .wav clips (~23.95 h total)
```

The dataset is licensed **CC BY-NC-SA 4.0**; follow its terms. Update the paths in the notebook's `CFG` dict.
