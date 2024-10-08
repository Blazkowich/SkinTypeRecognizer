from face_detection import are_eyes_detected, calculate_distance
from prediction import predict_image
import numpy as np
import cv2
import time
from load_gif import load_gif_frames
from PIL import Image


def analyze_skin_type():
    global display_time
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prediction_text = None
    analysis_done = False

    gif_frames, frame_duration = load_gif_frames(r'img/scan_anim.gif')
    face_percentage_img = cv2.imread(r'img/faces_percentage.png')

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

                    if 15 <= distance <= 25:
                        start_time = time.time()
                        while (time.time() - start_time) < 7:
                            for gif_frame in gif_frames:
                                gif_frame_resized = cv2.resize(gif_frame, (new_width, new_height))
                                gif_frame_padded = cv2.copyMakeBorder(
                                    gif_frame_resized,
                                    top=(window_height - new_height) // 2,
                                    bottom=(window_height - new_height + 1) // 2,
                                    left=(window_width - new_width) // 2,
                                    right=(window_width - new_width + 1) // 2,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0)
                                )
                                cv2.imshow('Webcam', gif_frame_padded)
                                if cv2.waitKey(frame_duration) & 0xFF == ord('q'):
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return

                        prediction = predict_image(frame)
                        normal_skin, oily_skin, dry_skin = prediction

                        prediction_text = f"         {normal_skin:.2%}\n\n\n\n\n\n\n         {dry_skin:.2%}\n\n\n\n\n\n\n         {oily_skin:.2%}"
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
                face_img_resized = cv2.resize(face_percentage_img, (new_width, new_height))
                face_img_padded = cv2.copyMakeBorder(
                    face_img_resized,
                    top=(window_height - new_height) // 2,
                    bottom=(window_height - new_height + 1) // 2,
                    left=(window_width - new_width) // 2,
                    right=(window_width - new_width + 1) // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )

                lines = prediction_text.split('\n')
                font_scale = 1.5
                font_thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Calculate total text height
                text_height = 0
                for line in lines:
                    text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                    text_height += text_size[1] + 10  # Adding some space between lines

                # Start position for the first line
                y0 = (window_height - text_height) // 2

                for i, line in enumerate(lines):
                    text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                    text_x = (window_width - text_size[0]) // 2
                    text_y = y0 + (i + 1) * text_size[1] + i * 10

                    cv2.putText(face_img_padded, line, (text_x, text_y), font, font_scale,
                                (1, 1, 1), font_thickness, cv2.LINE_AA)

                # Display the final image for 15 seconds
                start_display_time = time.time()
                while (time.time() - start_display_time) < 15:
                    cv2.imshow('Webcam', face_img_padded)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                analysis_done = False  # Reset for next analysis

        cv2.imshow('Webcam', frame_padded)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_skin_type()
