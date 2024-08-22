import time
import cv2
from src.prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
from concurrent.futures import ThreadPoolExecutor
from global_data import global_prediction_results, results_lock


class DistanceAndPrediction:
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.detector = FaceMeshDetector(maxFaces=1)
        self.W = 6.3
        self.f = 1500
        self.face_detected = False
        self.frame_count = 0

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
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return

        distance = self.calculate_distance(frame)
        if distance and 10 <= distance <= 30:
            self.face_detected = True
            self.executor.submit(self.predict_and_store, frame)
        else:
            if self.face_detected:
                self.reset_prediction_results()
                self.face_detected = False

    def predict_and_store(self, frame):
        """Predict skin type and store results."""
        predictions = predict_image(frame)
        formatted_results = self.format_prediction_results(predictions)

        with results_lock:
            if not self.face_detected:
                global_prediction_results.clear()
                print("Old data removed\n")
            global_prediction_results.extend(formatted_results)
            self.result_queue.put(formatted_results)

    def reset_prediction_results(self):
        """Reset the global prediction results to 0%."""
        with results_lock:
            global_prediction_results[:] = [
                "Normal: 0.00%",
                "Oily: 0.00%",
                "Dry: 0.00%",
                "Combination: 0.00%"
            ]
            self.result_queue.put(global_prediction_results)

    @staticmethod
    def format_prediction_results(predictions, duration=2, interval=0.2):
        """Format and return prediction results as text."""
        start_time = time.time()
        combined_skin_list = []
        normal_skin_list = []
        dry_skin_list = []
        oily_skin_list = []

        while time.time() - start_time < duration:
            if 'combination' in predictions:
                combined_skin_list.append(predictions['combination'])
            if 'normal' in predictions:
                normal_skin_list.append(predictions['normal'])
            if 'dry' in predictions:
                dry_skin_list.append(predictions['dry'])
            if 'oily' in predictions:
                oily_skin_list.append(predictions['oily'])

            time.sleep(interval)

        avg_combined = sum(combined_skin_list) / len(combined_skin_list) if combined_skin_list else 0
        avg_normal = sum(normal_skin_list) / len(normal_skin_list) if normal_skin_list else 0
        avg_dry = sum(dry_skin_list) / len(dry_skin_list) if dry_skin_list else 0
        avg_oily = sum(oily_skin_list) / len(oily_skin_list) if oily_skin_list else 0

        total = avg_normal + avg_oily + avg_dry + avg_combined
        if total > 0:
            avg_normal = (avg_normal / total) * 100
            avg_oily = (avg_oily / total) * 100
            avg_dry = (avg_dry / total) * 100
            avg_combined = (avg_combined / total) * 100

            return (f"Normal: {avg_normal:.2f}%",
                    f"Oily: {avg_oily:.2f}%",
                    f"Dry: {avg_dry:.2f}%",
                    f"Combination: {avg_combined:.2f}%")
        else:
            return "Normal: 0.00%", "Oily: 0.00%", "Dry: 0.00%", "Combination: 0.00%"
