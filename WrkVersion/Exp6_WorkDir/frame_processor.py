import multiprocessing
import cv2
from distance_and_prediction import DistanceAndPrediction


def resize_window_aspect_ratio(window_name, width, height):
    """Resize window to maintain a 9:16 aspect ratio."""
    aspect_ratio = 9 / 16
    new_height = int(width / aspect_ratio)
    if new_height != height:
        height = new_height
    cv2.resizeWindow(window_name, width, height)


def add_text_to_frame(frame, texts):
    """Add formatted text to the frame."""
    y_offset = 30
    for text in texts:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_offset += 40


def frame_processing_worker(frame_queue, result_queue):
    """Process frames from the queue and update results in a separate process."""
    distance_and_prediction = DistanceAndPrediction(result_queue)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        distance_and_prediction.process_frame(frame)


def start_processing(frame_queue, result_queue):
    """Start the frame processing in a separate process."""
    p = multiprocessing.Process(target=frame_processing_worker, args=(frame_queue, result_queue))
    p.start()
    return p
