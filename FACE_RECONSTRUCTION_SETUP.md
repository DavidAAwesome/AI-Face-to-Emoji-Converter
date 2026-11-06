# Face Reconstruction Patch

This patch adds a lightweight `face_reconstruction/` package and an updated
`avatar_generator.py`.

## Files added

- `face_reconstruction/__init__.py`
- `face_reconstruction/face_landmarks.py`
- `face_reconstruction/reconstruct_face.py`
- `avatar_generator.py` (updated to use the package)

## How to install

Copy the `face_reconstruction/` folder into the root of your project, next to:

- `app.py`
- `avatar_generator.py`
- `avatar_animator.py`
- `avatar_viewer.py`

Then replace your current `avatar_generator.py` with the updated one from this patch.

## Optional but recommended dependency

For better facial measurements, install MediaPipe:

```powershell
.\.venv\Scripts\python.exe -m pip install mediapipe
```

Without MediaPipe, the code still runs using OpenCV face detection, but the
personalization quality will be lower.

## What this improves

The older avatar generator mainly estimated color and used default face
proportions. This patch adds measured values for:

- face width
- face height
- eye spacing
- nose size
- mouth width
- approximate eye-to-mouth spacing
- eyebrow height

These values are saved into `output/avatar.json`, which `avatar_viewer.py`
reads to build the 3D avatar.

## Limitation

This is still a stylized, single-image approximation. It is **not** a full
photorealistic 3D head reconstruction system.
