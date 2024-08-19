import cv2
import time
import cvzone
from src.prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
from src.average_percent import average_percentages

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


def analyze_skin_type():
    global predictions

    cap = cv2.VideoCapture(1)
    detector = FaceMeshDetector(maxFaces=1)

    # Set the window to be resizable
    window_name = 'Webcam'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set initial size to maintain a 9:16 aspect ratio
    initial_width = 540
    initial_height = 960
    cv2.resizeWindow(window_name, initial_width, initial_height)

    W = 6.3  # The actual width of the object (in cm)
    f = 1500  # Pre-calculated focal length

    while True:
        ret, frame = cap.read()
        if not ret:
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

                # Use the latest predictions to update the display text
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

                prediction_text = (f"C: {combined_skin:.2f}%-"
                                   f"N: {normal_skin:.2f}%-"
                                   f"D: {dry_skin:.2f}%-"
                                   f"O: {oily_skin:.2f}%")

                print(prediction_text)

                # Display the prediction text on the frame
                cvzone.putTextRect(frame, prediction_text,
                                   (face[10][0] - 100, face[10][1] - 50),
                                   scale=2, colorR=(0, 255, 0))

                cv2.imshow(window_name, frame)

        cv2.imshow(window_name, frame)

        # Maintain the 9:16 aspect ratio even when resizing
        resize_window_aspect_ratio(window_name, initial_width, initial_height)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Collect and print average percentages
    avg_combined, avg_normal, avg_dry, avg_oily = average_percentages(predictions)
    avg_prediction_text = (f"Avg C: {avg_combined:.2f}%-"
                           f"N: {avg_normal:.2f}%-"
                           f"D: {avg_dry:.2f}%-"
                           f"O: {avg_oily:.2f}%")
    print(avg_prediction_text)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_skin_type()
