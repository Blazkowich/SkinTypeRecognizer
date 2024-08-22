import pytest
import cv2

def test_rtsp(ip, email, password):
    url = f'rtsp://{email}:{password}@{ip}:554/'  # Use port 554 by default
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        pytest.fail(f"Failed to connect to {url}")

    ret, frame = cap.read()
    if not ret:
        pytest.fail(f"Failed to read frame from {url}")

    cap.release()

@pytest.mark.parametrize("ip,email,password", [
    ("192.168.100.7", "gotogeoteam@gmail.com", "Otar#1996"),
])
def test_rtsp_connection(ip, email, password):
    test_rtsp(ip, email, password)
