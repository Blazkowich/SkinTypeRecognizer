import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model_path = r'../src/train/skin_type_model.h5'
img_width, img_height = 150, 150

model = load_model(model_path)

base_output_dir = 'captured_images'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)


def predict_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return prediction


def capture_and_predict():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    total_captures = 60
    capture_count = 0
    directory_count = 1

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

        cv2.imshow('Webcam', frame_padded)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Capturing images...")

            session_output_dir = os.path.join(base_output_dir, f'C{directory_count}')
            if not os.path.exists(session_output_dir):
                os.makedirs(session_output_dir)

            skin_type_counts = {'dry': 0, 'oily': 0, 'normal': 0}

            while capture_count < total_captures:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break

                prediction = predict_image(frame)
                class_idx = np.argmax(prediction[0])
                class_names = ['dry', 'oily', 'normal']
                predicted_class = class_names[class_idx]
                skin_type_counts[predicted_class] += 1

                type_folder = os.path.join(session_output_dir, predicted_class[0].upper())
                if not os.path.exists(type_folder):
                    os.makedirs(type_folder)
                filename = os.path.join(type_folder, f'Image_{capture_count + 1}.jpg')
                cv2.imwrite(filename, frame)

                print(
                    f'Image {capture_count + 1}: {predicted_class.capitalize()}: {prediction[0][class_idx] * 100:.2f}%')

                capture_count += 1

            print("\nSummary:")
            summary = []
            for skin_type, count in skin_type_counts.items():
                percentage = (count / total_captures) * 100
                summary.append(f'{skin_type[0].upper()}_{percentage:.2f}')
                print(f'{skin_type.capitalize()}: {percentage:.2f}%')

            summary_filename = '_'.join(summary)
            summary_filepath = os.path.join(session_output_dir, f'{summary_filename}.jpg')
            cv2.imwrite(summary_filepath, frame)

            capture_count = 0
            directory_count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


capture_and_predict()
