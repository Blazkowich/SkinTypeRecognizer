from face_detection import are_eyes_detected, calculate_distance
from prediction import predict_image
import numpy as np
import cv2
import time
from load_gif import load_gif_frames


def analyze_skin_type():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prediction_text = None
    display_time = 0
    analysis_done = False

    gif_frames, frame_duration = load_gif_frames(r'../scan_anim.gif')

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
                        start_time = time.time()
                        while (time.time() - start_time) < 7:
                            for gif_frame in gif_frames:
                                # noinspection PyUnusedLocal
                                frame_copy = frame_padded.copy()
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


if __name__ == "__main__":
    analyze_skin_type()
