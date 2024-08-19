# main.py
import cv2
import multiprocessing
from global_data import global_prediction_results
from frame_processor import start_processing


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


def main():
    vs = cv2.VideoCapture(1)

    frame_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    process = start_processing(frame_queue, result_queue)

    window_name = 'Webcam'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width = 540
    initial_height = 960
    resize_window_aspect_ratio(window_name, initial_width, initial_height)

    while True:
        grabbed, frame = vs.read()
        if grabbed and frame is not None:
            frame = cv2.flip(frame, 1)

            frame_queue.put(frame)

            if not result_queue.empty():
                global_prediction_results[:] = result_queue.get()

            add_text_to_frame(frame, global_prediction_results)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    frame_queue.put(None)
    process.join()

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
