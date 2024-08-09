import cv2
import numpy as np
from PIL import Image


def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy()
            frames.append(cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2BGR))
            gif.seek(len(frames))
    except EOFError:
        pass
    return frames, gif.info['duration']
