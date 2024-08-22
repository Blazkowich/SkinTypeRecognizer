import cv2
import multiprocessing
import pytest
import os
import time
from WrkVersion.Exp6_WorkDir.global_data import global_prediction_results
from WrkVersion.Exp6_WorkDir.frame_processor import start_processing, add_text_to_frame, resize_window_aspect_ratio
from WrkVersion.Exp6_WorkDir.distance_and_prediction import DistanceAndPrediction
from src.prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
from concurrent.futures import ThreadPoolExecutor


# Mocked or Dummy Implementation for Testing
def mock_predict_image(frame):
    # Dummy prediction implementation for testing
    return {
        'normal': 0.3,
        'dry': 0.5,
        'oily': 0.2,
        'combination': 0.0
    }


def mock_resize_window_aspect_ratio(window_name, width, height):
    """Mock function to replace actual resizing in tests."""
    pass


# Dummy DistanceAndPrediction for Testing
class MockDistanceAndPrediction:
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.detector = FaceMeshDetector(maxFaces=1)
        self.W = 6.3
        self.f = 1500
        self.face_detected = False
        self.frame_count = 0

    def calculate_distance(self, frame):
        return 20  # Mocked distance

    def process_frame(self, frame):
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
        predictions = mock_predict_image(frame)
        formatted_results = self.format_prediction_results(predictions)
        global_prediction_results[:] = formatted_results
        self.result_queue.put(formatted_results)

    def reset_prediction_results(self):
        global_prediction_results[:] = [
            "Normal: 0.00%",
            "Oily: 0.00%",
            "Dry: 0.00%",
            "Combination: 0.00%"
        ]
        self.result_queue.put(global_prediction_results)

    @staticmethod
    def format_prediction_results(predictions, duration=2, interval=0.2):
        avg_normal = predictions.get('normal', 0) * 100
        avg_oily = predictions.get('oily', 0) * 100
        avg_dry = predictions.get('dry', 0) * 100
        avg_combined = predictions.get('combination', 0) * 100

        return (
            f"Normal: {avg_normal:.2f}%",
            f"Oily: {avg_oily:.2f}%",
            f"Dry: {avg_dry:.2f}%",
            f"Combination: {avg_combined:.2f}%"
        )


@pytest.fixture
def image_paths():
    # Adjust the directory path as needed
    image_dir = r'D:\Nugzar\Oily-Dry-Skin-Types\test\dry'  # Directory containing your test images
    return [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]


def test_image_predictions(image_paths):
    frame_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Use the mock DistanceAndPrediction class for testing
    process = start_processing(frame_queue, result_queue)
    global_distance_and_prediction = MockDistanceAndPrediction(result_queue)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to open image file {image_path}.")
            continue

        frame_queue.put(image)

        # Allow time for the processing to complete
        time.sleep(0.5)  # Increased wait time for debugging

        if not result_queue.empty():
            result = result_queue.get()
            print(f"Raw Result from Queue: {result}")  # Debugging output
            global_prediction_results[:] = result
            print(f"Updated Global Prediction Results: {global_prediction_results}")

    frame_queue.put(None)
    process.join()

    # Check the test condition
    print(f"Final Global Prediction Results: {global_prediction_results}")
    dry_percentages = [
        float(result.split(': ')[1].replace('%', '')) for result in global_prediction_results if 'Dry' in result
    ]

    if not dry_percentages:
        print("No 'Dry' results found.")
        assert False, "Test Failed: No 'Dry' results found."

    for dry_percentage in dry_percentages:
        print(f"Dry Percentage: {dry_percentage}")
        assert dry_percentage > 90, f"Test Failed: 'Dry' is {dry_percentage:.2f}% which is not higher than 90%."

    print("Test Passed: 'Dry' is higher than 90%.")
