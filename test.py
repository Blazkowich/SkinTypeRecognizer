import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

model_path = r'train\skin_type_model.h5'
img_width, img_height = 150, 150
model = load_model(model_path)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

FOCAL_LENGTH = 600
EYE_HEIGHT_AT_1M = 5


def calculate_distance(eye_height):
    return (EYE_HEIGHT_AT_1M * FOCAL_LENGTH) / eye_height


def predict_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)[0]
    return prediction


def are_eyes_detected(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return eyes


def analyze_skin_type():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prediction_text = None
    display_time = 0
    analysis_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]
        window_width, window_height = cv2.getWindowImageRect('Webcam')[2:]
        aspect_ratio = frame_width / frame_height

        if window_width / window_height > aspect_ratio:
            new_width = int(window_height * aspect_ratio)
            new_height = window_height
        else:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)

        frame_resized = cv2.resize(frame, (new_width, new_height))
        frame_padded = cv2.copyMakeBorder(
            frame_resized,
            top=(window_height - new_height) // 2,
            bottom=(window_height - new_height + 1) // 2,
            left=(window_width - new_width) // 2,
            right=(window_width - new_width + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        if not analysis_done:
            eyes = are_eyes_detected(frame)
            if len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes:
                    eye_height_at_camera = eh
                    distance = calculate_distance(eye_height_at_camera)

                    if 10 <= distance <= 15:
                        for i in range(3, 0, -1):
                            countdown_text = f"{i}..."
                            text_size, _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                           2, 3)
                            text_x = (window_width - text_size[0]) // 2
                            text_y = (window_height + text_size[1]) // 2
                            frame_copy = frame_padded.copy()
                            cv2.putText(frame_copy, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        2, (255, 255, 255), 3, cv2.LINE_AA)
                            cv2.imshow('Webcam', frame_copy)
                            cv2.waitKey(1000)  # 1 second delay

                        for i in range(15):
                            dots = '.' * (i % 3 + 1)
                            analyzing_text = f"Analyzing{dots}"
                            text_size, _ = cv2.getTextSize(analyzing_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                           1, 2)
                            text_x = (window_width - text_size[0]) // 2
                            text_y = (window_height + text_size[1]) // 2
                            frame_copy = frame_padded.copy()
                            cv2.putText(frame_copy, analyzing_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Webcam', frame_copy)
                            if cv2.waitKey(333) & 0xFF == ord('q'):
                                cap.release()
                                cv2.destroyAllWindows()
                                return

                        prediction = predict_image(frame)
                        normal_skin, oily_skin, dry_skin = prediction

                        prediction_text = f"Normal: {normal_skin:.2%}, Oily: {oily_skin:.2%}, Dry: {dry_skin:.2%}"
                        display_time = time.time()
                        analysis_done = True
                        break
                else:
                    cv2.putText(frame_padded, 'Eyes detected but not in range.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_padded, 'No eyes detected. Please adjust.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
        else:
            if prediction_text and (time.time() - display_time) <= 10:
                text_size, _ = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = (window_width - text_size[0]) // 2
                text_y = (window_height + text_size[1]) // 2
                cv2.putText(frame_padded, prediction_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                analysis_done = False

        cv2.imshow('Webcam', frame_padded)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


analyze_skin_type()
