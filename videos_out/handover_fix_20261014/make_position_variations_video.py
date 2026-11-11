"""Join controlled reset-variation rollouts with role and outcome labels."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent / "position_variations_4426924"
ROWS = json.loads((HERE / "episodes.json").read_text())["episodes"]
ORDER = [row["video"] for row in ROWS]
LOOKUP = {row["video"]: row for row in ROWS}
WIDTH, HEIGHT, FPS = 1280, 720, 30
TEMP = HERE / "position_variations_temp.mp4"
OUTPUT = HERE / "position_variations_273920000.mp4"

writer = cv2.VideoWriter(str(TEMP), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
if not writer.isOpened():
    raise RuntimeError("Could not open MP4 writer")


def put_lines(frame, lines, top=65, size=1.0):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (42, top + i * 53), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA)


for episode, name in enumerate(ORDER, start=1):
    row = LOOKUP[name]
    giver = "UR5e" if row["giver_slot"] == 0 else "GEN3"
    receiver = "GEN3" if row["giver_slot"] == 0 else "UR5e"
    policy = "deterministic" if row["deterministic"] else "stochastic"
    case = row["case"].replace("_", " ")
    result = "SUCCESS" if row["success"] else f"NO SUCCESS (stage {row['peak_stage']})"
    title = f"{giver} gives -> {receiver} receives | {result}"
    subtitle = f"Start: {case} | Level 4 | 273.92M checkpoint"

    card = np.full((HEIGHT, WIDTH, 3), (33, 36, 40), dtype=np.uint8)
    put_lines(card, [title], top=310, size=0.95)
    put_lines(card, [subtitle], top=370, size=0.72)
    for _ in range(round(FPS * 0.75)):
        writer.write(card)

    cap = cv2.VideoCapture(str(HERE / name))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {name}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise RuntimeError(f"Invalid frame rate for {name}")
    repeats_due = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (HEIGHT, WIDTH):
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.rectangle(frame, (0, 0), (WIDTH, 99), (27, 30, 34), -1)
        put_lines(frame, [title], top=38, size=0.68)
        put_lines(frame, [subtitle], top=82, size=0.58)
        repeats_due += FPS * 4 / source_fps
        while repeats_due >= 1:
            writer.write(frame)
            repeats_due -= 1
    cap.release()

writer.release()
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-i", str(TEMP), "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", str(OUTPUT)],
    check=True,
)
TEMP.unlink()
print(OUTPUT)
