# global_data.py
import threading

global_prediction_results = []
results_lock = threading.Lock()
