import cv2
from threading import Thread
from queue import Queue
from distance_and_prediction import DistanceAndPrediction
from global_data import global_prediction_results

class WebcamVideoStream:
    def __init__(self, src=1):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Consider reducing resolution
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Consider reducing resolution
        self.frame_queue = Queue(maxsize=10)  # Buffer up to 10 frames
        self.stopped = False

    def start(self):
        # Allow some time for the camera to warm up
        self.warm_up_camera()
        self.dp = DistanceAndPrediction()
        Thread(target=self.update, args=(), daemon=True).start()
        Thread(target=self.process_frames, args=(), daemon=True).start()
        return self

    def warm_up_camera(self):
        print("Warming up camera...")
        for _ in range(30):  # Warm up for 30 frames
            self.stream.read()
        print("Camera warmed up.")

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

    def process_frames(self):
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.dp.process_frame(frame)

    def read(self):
        return self.frame_queue.queue[-1] if not self.frame_queue.empty() else None

    def stop(self):
        self.stopped = True
        self.stream.release()
