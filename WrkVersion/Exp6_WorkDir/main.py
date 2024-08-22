# main.py
import cv2
import multiprocessing
from global_data import global_prediction_results
from frame_processor import start_processing, add_text_to_frame, resize_window_aspect_ratio


def main():
    vs = cv2.VideoCapture(0)

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
            #frame = cv2.flip(frame, 1)

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
