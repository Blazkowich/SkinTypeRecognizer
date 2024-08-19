import multiprocessing
from distance_and_prediction import DistanceAndPrediction


def frame_processing_worker(frame_queue, result_queue):
    """Process frames from the queue and update results in a separate process."""
    distance_and_prediction = DistanceAndPrediction(result_queue)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        distance_and_prediction.process_frame(frame)


def start_processing(frame_queue, result_queue):
    """Start the frame processing in a separate process."""
    p = multiprocessing.Process(target=frame_processing_worker, args=(frame_queue, result_queue))
    p.start()
    return p
