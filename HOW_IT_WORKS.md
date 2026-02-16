# Virtual Board — How It Works

## What Is This?
A **Virtual Board** application that lets you draw, erase, and place 3D shapes in the air using just your hand gestures — no mouse or keyboard needed.

## Technologies Used
| Library | Purpose |
|---------|---------|
| **OpenCV** | Camera input, drawing on screen, displaying the UI |
| **MediaPipe** | Detects hand landmarks (21 points on each hand) |
| **NumPy** | Creates and manipulates the drawing canvas |

## How It Works (Step by Step)

1. **Camera captures a frame** → flipped horizontally so it feels like a mirror.
2. **MediaPipe detects the hand** → finds 21 landmark points (fingertips, knuckles, wrist, etc.).
3. **We check which fingers are up** → by comparing fingertip positions with knuckle positions (if tip is higher than knuckle, finger is "up").
4. **Based on the finger combination, we decide the mode:**

| Gesture | What Happens |
|---------|-------------|
| ☝️ Index finger only | **Draw** on the canvas |
| ✌️ Index + Middle finger | **Selection mode** — pick colors from the top bar |
| 🤟 Index + Middle + Ring | **Shape mode** — select & manipulate 3D shapes |
| ✊ Fist (no fingers up) | **Place** the current 3D shape on canvas |
| 🖐️ All fingers open (one hand) | **Erase** — clears the entire board |
| 🙌 Both hands open | Shows **"Thanks Mam — From Anuj & Sachin Kumar Jha"** |

5. **Drawing** works by connecting the previous finger position to the current one with a line — frame by frame, it creates a smooth stroke.
6. **The canvas (black image) is merged with the camera feed** using bitwise operations so drawings appear on top of the live video.

## Key Concepts

- **Landmark Detection**: MediaPipe gives us 21 (x, y) coordinates per hand. We use specific points like index tip (point 8), middle tip (point 12), etc.
- **Finger Up Detection**: A finger is "up" if its tip landmark is above (lower y-value) its PIP joint landmark.
- **Canvas Merging**: We keep a separate black canvas for drawings and overlay it on the camera image every frame using `cv2.bitwise_and` and `cv2.bitwise_or`.

## How to Run
```
pip install opencv-python mediapipe numpy
python virtual_board.py
```
Press **'q'** to quit.
