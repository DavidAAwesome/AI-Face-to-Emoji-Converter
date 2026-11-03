# AI-Face-to-Emoji-Converter

A real-time computer vision system that detects facial expressions using deep learning and overlays corresponding emojis on the user's face.

CSI 571 project by David Aduku and DeAndre Lawson, University at Albany. The design follows the project proposal, *Facial Expression to Emoji Conversion Using Deep Learning*.

```
Webcam / Image → Face Detection → Preprocessing → CNN Expression Classifier
              → Expression Label → Emoji Mapping → Emoji Overlay
```

| Expression | Emoji | | Expression | Emoji |
|---|---|---|---|---|
| happy | 😀 | | fear | 😨 |
| sad | 🙁 | | disgust | 🤢 |
| angry | 😠 | | neutral | 😐 |
| surprise | 😮 | | *uncertain* (below the confidence threshold) | 🤔 |

The overlay uses PNG emoji from [assets/emojis/](assets/emojis/) instead of Unicode characters, so the emoji look the same on every platform (proposal Sec. III-D).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`opencv-python` is pinned below 5.0 because OpenCV 5 removed the Haar cascade face detector.

### Dataset (FER2013)

Download either version into `data/`:

* **Images:** [Kaggle `msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013), unzipped to `data/fer2013/train/<class>/*.png` and `data/fer2013/test/<class>/*.png`. 10% of `train` is held out for validation.
* **CSV:** `fer2013.csv` from the [Kaggle FER challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), saved as `data/fer2013.csv`. Its `Training`, `PublicTest` and `PrivateTest` rows become the train, validation and test splits.

The test split is never used for training or model selection.

## Usage

### 1. Train

```bash
python train.py --data data/fer2013.csv                 # baseline CNN -> checkpoints/best_model.pt
python train.py --data data/fer2013 --arch resnet18 --lr 3e-4 --out checkpoints/resnet18.pt
```

Useful options: `--no-augment`, `--balance {weights,sampler,none}` (FER2013 has about 16× fewer *disgust* images than *happy* ones), `--merge disgust=angry` (change the class set), `--epochs`, and `--limit N` for quick tests. The best epoch by validation accuracy is saved, and training stops early when validation accuracy stops improving.

Training the baseline CNN takes a few minutes per epoch on a CPU and much less on a CUDA GPU, which is used automatically if present.

### 2. Evaluate

```bash
python evaluate.py --data data/fer2013.csv [--checkpoint checkpoints/best_model.pt] [--tta]
```

This writes the following to `results/<checkpoint name>/`:

* `metrics.json`: accuracy, per-class precision, recall and F1, macro and weighted averages, and the most frequent confusions.
* `confusion_matrix.png`: raw counts and a row-normalised version.
* `threshold_sweep.png/json`: coverage vs. accuracy for each confidence threshold.
* `robustness.png/json`: test accuracy under dark, bright and low-contrast images, blur, noise, 15°/30° rotation, eye and mouth occlusion, low resolution (distance), and mirroring.
* `training_curves.png`: loss and accuracy per epoch.

`--tta` averages each prediction with the prediction for the mirrored face.

### 3. Run the emoji converter

```bash
python app.py                                   # live webcam
python app.py --image me.jpg friends.png        # still images -> results/demo/*_emoji.png
python app.py --video clip.mp4 --save results/clip_emoji.mp4
```

Each detected face gets its own emoji. Faces are tracked across frames and their probabilities are smoothed so the emoji does not flicker. A side panel shows the main face's emoji, expression, confidence score and all class probabilities, and an FPS and latency readout runs along the top of the video.

| Key | Action |
|---|---|
| `m` | overlay mode: **mask** (emoji covers the face) → **side** (emoji beside a box) → **off** |
| `+` / `-` | raise / lower the confidence threshold |
| `p` | toggle the side panel |
| `s` | save a snapshot to `results/snapshots/` |
| `q` / `Esc` | quit |

Options: `--threshold 0.4`, `--smoothing 0.6` (1 = no smoothing), `--detector yunet`, `--tta`, `--emoji-dir`.

**Custom emoji:** put your own RGBA `<expression>.png` files (for example `happy.png` and `uncertain.png`) in a folder and pass it with `--emoji-dir my_pack/`. Any expression without an image falls back to the built-in emoji.

**More robust face detection:** the default Haar cascade needs no download but works best on frontal faces. YuNet, a CNN detector, copes better with head pose and lighting:

```bash
python -m emoji_converter.face_detector --download
python app.py --detector yunet
```

### 4. Benchmark real-time performance

```bash
python benchmark.py --frames 300                # webcam: detection / classification / end-to-end ms, FPS
python benchmark.py --video clip.mp4 --detector yunet
python benchmark.py --model-only checkpoints/*.pt   # per-face inference time and parameter count
```

### 5. Run all training experiments

```bash
python run_experiments.py --data data/fer2013.csv [--epochs 30] [--only baseline no_augment]
```

This trains, evaluates and times each configuration from proposal Sec. IV-B and writes a comparison table to `results/experiments.md`:

| Experiment | What it tests |
|---|---|
| `baseline` | VGG-style CNN trained from scratch |
| `small_cnn` | a smaller, faster architecture |
| `resnet18_transfer` / `resnet18_scratch` | transfer learning vs. training from scratch |
| `no_augment` | the effect of data augmentation |
| `balanced_sampler` | class-balanced sampling instead of a weighted loss |
| `merge_disgust` | a 6-class set with *disgust* merged into *angry* |

## Project layout

```
emoji_converter/
  config.py          classes, emoji mapping, paths
  data.py            FER2013 loading (CSV or folders), splits, augmentation, preprocessing
  model.py           EmotionCNN, SmallCNN, ResNet-18 transfer model, checkpoint save/load
  face_detector.py   Haar cascade / YuNet detection, face cropping
  pipeline.py        detect → classify → threshold → per-face tracking and smoothing
  emoji_assets.py    procedurally drawn emoji PNGs, custom emoji loading
  visualize.py       emoji overlay, labels, side panel, HUD
train.py  evaluate.py  app.py  benchmark.py  run_experiments.py
tests/test_pipeline.py   smoke tests (no dataset needed): python tests/test_pipeline.py
```

## Notes and known limitations

* FER2013 images are 48×48 grayscale with noisy labels. Published CNN results are usually around 65–73% test accuracy, and human agreement on the dataset is roughly 65%, so expect visually similar expressions (fear/sad/surprise, angry/disgust) to be confused.
* Webcam faces differ from FER2013 crops in lighting, pose and resolution. `evaluate.py` measures robustness to these conditions synthetically, and live tests across different people and settings complete the robustness experiments (Sec. IV-E).
* Expressions are not always discrete. The confidence threshold shows 🤔 instead of guessing when the model is unsure.
