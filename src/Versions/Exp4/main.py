from data_capture import WebcamVideoStream
from display import main as display_main
import threading


def run_data_capture_thread():
    capture_thread = threading.Thread(target=lambda: WebcamVideoStream(src=1).start(), daemon=True)
    capture_thread.start()
    return capture_thread


def main():
    # Start data capture thread
    _ = run_data_capture_thread()

    # Run the display
    display_main()


if __name__ == "__main__":
    main()
