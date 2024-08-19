import cv2
from concurrent.futures import ThreadPoolExecutor
from src.prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
from threading import Thread
import queue

# Shared variables
predictions = {}
frame_queue = queue.Queue(maxsize=1)  # Buffer to hold one frame at a time

def predict_skin_type(frame):
    """Predict skin type for a given frame."""
    global predictions
    predictions = predict_image(frame)

def resize_window_aspect_ratio(window_name, width, height):
    """Resize window to maintain a 9:16 aspect ratio."""
    aspect_ratio = 9 / 16
    new_height = int(width / aspect_ratio)
    if new_height != height:
        height = new_height
    cv2.resizeWindow(window_name, width, height)

def format_prediction_results():
    """Format and return prediction results as text."""
    global predictions
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

        return (f"Normal: {normal_skin:.2f}%",
                f"Oily: {oily_skin:.2f}%",
                f"Dry: {dry_skin:.2f}%",
                f"Combination: {combined_skin:.2f}%")
    else:
        return ("Normal: 0.00%", "Oily: 0.00%", "Dry: 0.00%", "Combination: 0.00%")

def add_text_to_frame(frame, texts):
    """Add formatted text to the frame."""
    y_offset = 30
    for text in texts:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_offset += 40

def async_predict_skin_type(frame):
    """Asynchronous wrapper for skin type prediction."""
    global predictions
    predictions = predict_image(frame)

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

def process_frame():
    """Process frames to perform predictions and update the video display."""
    global predictions

    vs = WebcamVideoStream(src=1).start()
    detector = FaceMeshDetector(maxFaces=1)

    window_name = 'Webcam'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width = 540
    initial_height = 960
    resize_window_aspect_ratio(window_name, initial_width, initial_height)

    W = 6.3  # The actual width of the object (in cm)
    f = 1500  # Pre-calculated focal length

    frame_skip = 3  # Process every 3rd frame
    frame_count = 0

    executor = ThreadPoolExecutor(max_workers=2)

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.flip(frame, 1)
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            frame, faces = detector.findFaceMesh(frame, draw=False)

            if faces:
                face = faces[0]
                pointLeft = face[374]
                pointRight = face[145]

                w, _ = detector.findDistance(pointLeft, pointRight)
                distance = (W * f) / w
                distance = int(distance)
                print(f"Distance: {distance} cm")

                if 20 <= distance <= 30:
                    future = executor.submit(async_predict_skin_type, frame)
                    future.result()

                    texts = format_prediction_results()
                    add_text_to_frame(frame, texts)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()
