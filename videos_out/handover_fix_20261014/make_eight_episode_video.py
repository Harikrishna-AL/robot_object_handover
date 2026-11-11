"""Join eight alternating-role checkpoint rollouts with episode labels."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent / "multi_episode_4420579"
ROWS = json.loads((HERE / "episodes.json").read_text())["episodes"]
ORDER = [row["video"] for row in ROWS]
LOOKUP = {row["video"]: row for row in ROWS}
WIDTH, HEIGHT, FPS = 1280, 720, 30
TEMP = HERE / "role_switching_eight_episodes_temp.mp4"
OUTPUT = HERE / "role_switching_eight_episodes.mp4"

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
    title = f"Episode {episode}: {giver} gives -> {receiver} receives"
    subtitle = f"Curriculum level 4  |  {policy} policy  |  Stage {row['peak_stage']} reached"

    card = np.full((HEIGHT, WIDTH, 3), (33, 36, 40), dtype=np.uint8)
    put_lines(card, [title, subtitle], top=310, size=1.0)
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
        put_lines(frame, [title, subtitle], top=38, size=0.73)
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
