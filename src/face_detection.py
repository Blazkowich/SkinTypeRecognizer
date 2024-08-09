import cv2

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

FOCAL_LENGTH = 600
EYE_HEIGHT_AT_1M = 5


def calculate_distance(eye_height):
    return (EYE_HEIGHT_AT_1M * FOCAL_LENGTH) / eye_height


def are_eyes_detected(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return eyes
