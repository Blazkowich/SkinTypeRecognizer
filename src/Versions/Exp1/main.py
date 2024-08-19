import cv2
from threading import Thread
import queue
from distance_and_prediction import DistanceAndPrediction

# Shared queue for passing prediction results
result_queue = queue.Queue(maxsize=1)


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not frame_queue.full():
                frame_queue.put(self.frame)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


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
    vs = WebcamVideoStream(src=1).start()
    dp = DistanceAndPrediction(result_queue)

    window_name = 'Webcam'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width = 540
    initial_height = 960
    resize_window_aspect_ratio(window_name, initial_width, initial_height)

    frame_skip = 3  # Process every 3rd frame
    frame_count = 0
    last_results = ("Normal: 0.00%", "Oily: 0.00%", "Dry: 0.00%", "Combination: 0.00%")  # Initial text

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.flip(frame, 1)
            frame_count += 1

            if frame_count % frame_skip == 0:
                dp.process_frame(frame)

            if not result_queue.empty():
                last_results = result_queue.get()

            add_text_to_frame(frame, last_results)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=1)  # Shared queue for frames
    main()
