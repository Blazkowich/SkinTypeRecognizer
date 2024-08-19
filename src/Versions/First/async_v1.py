import cv2
import threading
from queue import Queue
from src.prediction import predict_image
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone

# Shared variables and queue for communication between threads
predictions = {}
predictions_lock = threading.Lock()
frame_queue = Queue()
stop_event = threading.Event()

def predict_skin_type():
    global predictions
    while not stop_event.is_set():
        frame = frame_queue.get()
        if frame is None:
            break
        # Convert the frame to the required format for the prediction function
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Assuming predict_image works with RGB images
        with predictions_lock:
            predictions = predict_image(frame_rgb)
        frame_queue.task_done()

def capture_video():
    cap = cv2.VideoCapture(1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        frame = cv2.flip(frame, 1)
        frame_queue.put(frame)
    cap.release()

async def analyze_skin_type():
    global predictions
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

    # Start threads for video capture and prediction
    capture_thread = threading.Thread(target=capture_video, daemon=True)
    prediction_thread = threading.Thread(target=predict_skin_type, daemon=True)
    capture_thread.start()
    prediction_thread.start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
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
                    # Use the latest predictions to update the display text
                    with predictions_lock:
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
                        if faces:
                            cvzone.putTextRect(frame, prediction_text,
                                               (face[10][0] - 100, face[10][1] - 50),
                                               scale=2, colorR=(0, 255, 0))

            cv2.imshow(window_name, frame)

            # Maintain the 9:16 aspect ratio even when resizing
            aspect_ratio = 9 / 16
            new_height = int(initial_width / aspect_ratio)
            if new_height != initial_height:
                initial_height = new_height
            cv2.resizeWindow(window_name, initial_width, initial_height)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Stop threads
    stop_event.set()
    frame_queue.put(None)  # Signal the prediction thread to exit
    prediction_thread.join()
    capture_thread.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the analysis in an asynchronous event loop
    import asyncio
    asyncio.run(analyze_skin_type())
