import cv2
import multiprocessing
import sys
from global_data import global_prediction_results
from frame_processor import start_processing, add_text_to_frame, resize_window_aspect_ratio


def main(video_path=None):
    # Use a video file if provided, otherwise use the webcam
    vs = cv2.VideoCapture(video_path if video_path else 0)

    if not vs.isOpened():
        print("Error: Unable to open video source.")
        return

    frame_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    process = start_processing(frame_queue, result_queue)

    window_name = 'Video Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width = 2048
    initial_height = 1080
    resize_window_aspect_ratio(window_name, initial_width, initial_height)

    while True:
        grabbed, frame = vs.read()
        if not grabbed or frame is None:
            print("End of video file or failed to grab a frame.")
            break

        # frame = cv2.flip(frame, 1)  # Optional: Flip frame if necessary

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
    # Check for a video file path passed as an argument
    video_path = '../../Assets/oil-vid.mp4'
    main(video_path)
