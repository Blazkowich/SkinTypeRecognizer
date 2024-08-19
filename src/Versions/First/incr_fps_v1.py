import cv2
import time
import cvzone
from src.prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
from src.average_percent import average_percentages
from threading import Thread

# Shared variable to store prediction results
predictions = {}

def predict_skin_type(frame):
    global predictions
    predictions = predict_image(frame)

def resize_window_aspect_ratio(window_name, width, height):
    """Resize window to maintain a 9:16 aspect ratio."""
    aspect_ratio = 9 / 16
    new_height = int(width / aspect_ratio)
    if new_height != height:
        height = new_height
    cv2.resizeWindow(window_name, width, height)

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def analyze_skin_type():
    global predictions

    # Initialize webcam and face mesh detector
    vs = WebcamVideoStream(src=1).start()
    detector = FaceMeshDetector(maxFaces=1)

    # Set up window
    window_name = 'Webcam'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_width = 540
    initial_height = 960
    cv2.resizeWindow(window_name, initial_width, initial_height)

    W = 6.3  # The actual width of the object (in cm)
    f = 1500  # Pre-calculated focal length

    while True:
        frame = vs.read()
        if frame is None:
            print("Failed to capture image")
            break

        frame = cv2.flip(frame, 1)

        # Detect face mesh
        frame, faces = detector.findFaceMesh(frame, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[374]
            pointRight = face[145]

            # Calculate the distance from the camera
            w, _ = detector.findDistance(pointLeft, pointRight)
            distance = (W * f) / w
            distance = int(distance)
            print(f"Distance: {distance} cm")

            if 20 <= distance <= 30:
                predict_skin_type(frame)

                # Display prediction results
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

                    cv2.putText(frame, f"Normal: {normal_skin:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, f"Oily: {oily_skin:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Dry: {dry_skin:.2f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Combination: {combined_skin:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_skin_type()
