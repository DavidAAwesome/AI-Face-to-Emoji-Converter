# 3D avatar integration patch for AI-Face-to-Emoji-Converter

This patch uses the existing `emoji_converter.pipeline.ExpressionPipeline` and existing trained `.pt` checkpoint. It does **not** replace the recognition dataset, retrain the model, or change the evaluation scripts.

## Files to copy

Place the four `.py` files in this archive alongside the original `app.py`, `train.py`, and `requirements.txt` in your VS Code project. Back up your original `app.py` before replacing it. Do not delete the existing `emoji_converter/` package, trained checkpoint, or dataset.

## Installation

From your project directory in PowerShell (no environment activation needed):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install panda3d
```

If `.venv` doesn't exist, create it with `python -m venv .venv` first. Panda3D supports specific Python versions; choose a compatible interpreter.

## Webcam avatar from the first face

```powershell
.\.venv\Scripts\python.exe app.py --avatar --mode off
```

This displays the usual webcam preview plus a *separate* Panda3D avatar window. The first detected face is used to create `results/avatar.json` and subsequent classifier predictions drive expression presets. If `results/avatar.json` already exists it is reused.

## Use one photo to initialize avatar appearance while animating via webcam

```powershell
.\.venv\Scripts\python.exe app.py --avatar --avatar-photo "my_photo.jpg" --mode off
```

To generate a new avatar from another photo, use a different config path (`--avatar-config results/avatar_person2.json`) or delete the old JSON. The `--image` option in the existing app continues to produce 2D image outputs; it does not launch the Panda3D avatar mode.

## Important limitations

- The avatar generator uses a Haar detector and rough cheek/hair color samples; it does NOT reconstruct individual 3D anatomy or hair style with high likeness.
- Facial dimensions in the JSON currently default to `1.0`; edits are possible, but no learned identity-fitting model is included.
- Emotions drive simple facial presets, not detailed facial motion or head pose. To achieve precise live movement, add a MediaPipe-based tracking module later.
- Existing `emoji_converter/` source files and trained `.pt` checkpoint are required to run `app.py`; these weren't among the files uploaded for this patch.
- Multi-face recognition still works in the original overlay; the separate 3D viewer uses the highest-confidence detected face, not a persistent identity tracker.
- Panda3D's window event loop and the OpenCV event loop have not been tested on your Windows machine. If any GUI errors occur, share the traceback.

## Normal original behavior

```powershell
.\.venv\Scripts\python.exe app.py
```

Without `--avatar`, the original 2D emoji overlay continues normally.
