import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Define paths
model_path = 'train\skin_type_model.h5'
img_width, img_height = 150, 150

# Load the trained model
model = load_model(model_path)

# Create a base directory to save the captured images
base_output_dir = 'captured_images'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

def predict_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return prediction

def capture_and_predict():
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
    total_captures = 60
    capture_count = 0
    directory_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow('Webcam', frame)

        # Press 'c' to start capturing 60 images and analyze them
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print("Capturing images...")
            
            # Create a directory for the current capture session
            session_output_dir = os.path.join(base_output_dir, f'C{directory_count}')
            if not os.path.exists(session_output_dir):
                os.makedirs(session_output_dir)
                
            # Initialize skin type counts for this session
            skin_type_counts = {'dry': 0, 'oily': 0, 'normal': 0}

            # Capture 60 frames
            while capture_count < total_captures:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Predict the skin type
                prediction = predict_image(frame)
                class_idx = np.argmax(prediction[0])
                class_names = ['dry', 'oily', 'normal']
                predicted_class = class_names[class_idx]
                skin_type_counts[predicted_class] += 1
                
                # Save the captured frame in the corresponding folder
                type_folder = os.path.join(session_output_dir, predicted_class[0].upper())
                if not os.path.exists(type_folder):
                    os.makedirs(type_folder)
                filename = os.path.join(type_folder, f'Image_{capture_count + 1}.jpg')
                cv2.imwrite(filename, frame)
                
                print(f'Image {capture_count + 1}: {predicted_class.capitalize()}: {prediction[0][class_idx]*100:.2f}%')

                capture_count += 1

            # Calculate and display summary
            print("\nSummary:")
            summary = []
            for skin_type, count in skin_type_counts.items():
                percentage = (count / total_captures) * 100
                summary.append(f'{skin_type[0].upper()}_{percentage:.2f}')
                print(f'{skin_type.capitalize()}: {percentage:.2f}%')

            # Save the summary as a filename
            summary_filename = '_'.join(summary)
            summary_filepath = os.path.join(session_output_dir, f'{summary_filename}.jpg')
            cv2.imwrite(summary_filepath, frame)  # Save the last frame as a placeholder for summary

            # Reset for next session
            capture_count = 0
            directory_count += 1

        # Press 'q' to quit the webcam window
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start capturing and predicting
capture_and_predict()
