# VoiceProtect
Deepfake audio detection using spectral-subtraction denoising, Mel-spectrograms and a CNN-BiLSTM model.
# VoiceProtect: Deepfake Audio Detection

VoiceProtect is a deep learning system that classifies speech as **bonafide (real)** or **spoof (AI-generated / voice-converted)**. It targets the compressed, noisy audio that real-world deepfakes usually arrive in, so audio is denoised with **spectral subtraction** before it is converted to **Mel-spectrograms** and passed to a **CNN + BiLSTM** classifier.

## Highlights

- Spectral-subtraction denoising (STFT-based) to reduce background noise and codec artifacts before feature extraction
- 128x128 log-Mel-spectrogram features extracted with Librosa
- Hybrid architecture: 3 convolutional blocks for spatial features, then 2 Bidirectional LSTM layers for temporal patterns
- Evaluated with accuracy, precision/recall/F1, confusion matrix, ROC-AUC and Equal Error Rate (EER)

## Dataset

Built on a filtered subset of the **ASVspoof 2021 Deepfake (DF) evaluation set**, listed in `filter_auds.csv` (about 13,100 utterances).

| Property | Details |
|---|---|
| Labels | `1` = bonafide (4,377), `0` = spoof (8,754) |
| Source corpora | ASVspoof (5,144), VCC 2020 (4,620), VCC 2018 (3,367) |
| Compression codecs | mp3, m4a and ogg at low/high bitrates, plus uncompressed (9 conditions) |
| Spoof types | Traditional, neural autoregressive and neural non-autoregressive vocoders |

The audio files themselves are **not included** in this repo. Download the ASVspoof 2021 DF evaluation data and update the paths in the notebook.

## Pipeline

1. **Load** audio with Librosa at its native sampling rate
2. **Denoise** with spectral subtraction (noise profile estimated from the first frames, magnitude floored at zero, signal rebuilt with the inverse STFT)
3. **Extract** a 128-band Mel-spectrogram and convert it to dB
4. **Pad or crop** to a fixed 128x128 input
5. **Classify** with the CNN-BiLSTM and a sigmoid output
6. **Evaluate** on a 70/30 train/test split

## Model

```
Conv2D(32) -> MaxPool -> Dropout
Conv2D(64) -> MaxPool -> Dropout
Conv2D(128) -> MaxPool -> Dropout
TimeDistributed(Flatten)
BiLSTM(128) -> Dropout
BiLSTM(64) -> Dropout
Dense(256) -> Dropout -> Dense(1, sigmoid)
```

Adam optimizer, binary cross-entropy loss, 10 epochs.

## Results (this notebook)

| Metric | Value |
|---|---|
| Test accuracy | 82.35% |
| ROC-AUC | 0.827 |
| EER | 25.5% |
| Spoof recall | 0.95 |
| Bonafide recall | 0.58 |

The model is very good at catching spoofed audio (95% recall) but misses more real speech than ideal, which is a natural next area to improve.

## Tech Stack

Python, TensorFlow/Keras, Librosa, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn

## How to Run

```bash
pip install librosa tensorflow scikit-learn pandas numpy matplotlib seaborn
```

1. Download the ASVspoof 2021 DF evaluation audio
2. Set `audio_folder` and the `filter_auds.csv` path in the notebook
3. Run `denoisingwithspectralsub2_copy.ipynb` from top to bottom

## Future Work

- Add MFCC / LFCC features alongside Mel-spectrograms
- Balance the classes and tune the decision threshold to raise bonafide recall
- Try attention or transformer-based models
- Build a simple web demo for uploading and checking audio
