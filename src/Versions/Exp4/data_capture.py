import cv2
from threading import Thread, Lock
from global_data import global_prediction_results
from prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
from concurrent.futures import ThreadPoolExecutor
import time


class DistanceAndPrediction:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.detector = FaceMeshDetector(maxFaces=1)
        self.W = 6.3  # The actual width of the object (in cm)
        self.f = 1500  # Pre-calculated focal length

    def calculate_distance(self, frame):
        """Calculate distance from the face."""
        frame, faces = self.detector.findFaceMesh(frame, draw=False)
        if faces:
            face = faces[0]
            pointLeft = face[374]
            pointRight = face[145]
            w, _ = self.detector.findDistance(pointLeft, pointRight)
            distance = (self.W * self.f) / w
            return int(distance)
        return None

    def process_frame(self, frame):
        """Process the frame for distance and prediction."""
        distance = self.calculate_distance(frame)
        if distance and 20 <= distance <= 30:
            self.executor.submit(self.predict_and_store, frame)

    def predict_and_store(self, frame):
        """Predict skin type and store results."""
        predictions = predict_image(frame)
        formatted_results = self.format_prediction_results(predictions)
        global_prediction_results[:] = formatted_results  # Update global variable

    @staticmethod
    def format_prediction_results(predictions):
        """Format and return prediction results as text."""
        normal_skin = predictions.get('normal', 0)
        oily_skin = predictions.get('oily', 0)
        dry_skin = predictions.get('dry', 0)
        combined_skin = predictions.get('combination', 0)
        total = normal_skin + oily_skin + dry_skin + combined_skin
        if total > 0:
            normal_skin = (normal_skin / total) * 100
            oily_skin = (oily_skin / total) * 100
            dry_skin = (dry_skin / total) * 100
            combined_skin = (combined_skin / total) * 100

            final_results = (f"Normal: {normal_skin:.2f}%",
                             f"Oily: {oily_skin:.2f}%",
                             f"Dry: {dry_skin:.2f}%",
                             f"Combination: {combined_skin:.2f}%")
            return final_results
        else:
            return "Normal: 0.00%", "Oily: 0.00%", "Dry: 0.00%", "Combination: 0.00%"


class WebcamVideoStream:
    def __init__(self, src=1):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.frame = None
        self.stopped = False
        self.lock = Lock()
        self.dp = DistanceAndPrediction()
        self.last_processed_time = time.time()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
                current_time = time.time()
                if current_time - self.last_processed_time > 1:
                    # Process the frame every second to reduce processing load
                    self.dp.process_frame(frame)
                    self.last_processed_time = current_time

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
