import os
import cv2
import inference

# Set API key
os.environ["ROBOFLOW_API_KEY"] = "8Qs7F54MSyWFOmMq8Yva"


def predict_image(model, frame, num_predictions=10):
    """Perform multiple predictions on a single frame."""
    predictions_list = []
    for _ in range(num_predictions):
        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform prediction
        results = model.infer(image=image_rgb)
        predictions = results[0].predictions

        # Collect predictions
        predictions_list.append({label: prediction.confidence for label, prediction in predictions.items()})

    return predictions_list


def summarize_predictions(predictions_list):
    """Summarize predictions from multiple runs."""
    summary = {label: [] for label in predictions_list[0]}  # Initialize summary dictionary

    for predictions in predictions_list:
        for label, confidence in predictions.items():
            summary[label].append(confidence)

    summarized_results = {label: sum(confidences) / len(confidences) for label, confidences in summary.items()}
    return summarized_results


def summarize_all_predictions(all_summarized_results):
    """Summarize predictions from all frames in the video."""
    combined_summary = {label: 0.0 for label in all_summarized_results[0]}  # Initialize summary dictionary

    for summarized_results in all_summarized_results:
        for label, confidence in summarized_results.items():
            combined_summary[label] += confidence

    # Normalize the combined summary to ensure it adds up to 1 (or 100%)
    total_confidence = sum(combined_summary.values())
    overall_summary = {label: (confidence / total_confidence) for label, confidence in combined_summary.items()}
    return overall_summary


def process_video(video_path, output_file, num_predictions=10):
    """Process a video file and write overall predictions summary to a text file."""
    try:
        model = inference.get_roboflow_model("skin-type-tgow5/1")  # Load the model once

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        all_summarized_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Perform multiple predictions on the current frame
            predictions_list = predict_image(model, frame, num_predictions)

            # Summarize predictions for the current frame
            summarized_results = summarize_predictions(predictions_list)
            all_summarized_results.append(summarized_results)

        cap.release()

        # Summarize predictions across all frames
        overall_summary = summarize_all_predictions(all_summarized_results)

        # Format the overall summary
        formatted_overall_summary = "\n".join(
            [f"{label}: {confidence * 100:.2f}%" for label, confidence in overall_summary.items()])

        # Write the overall summary to the file
        with open(output_file, 'w') as file:
            file.write("Overall Summary of Predictions:\n")
            file.write(f"{formatted_overall_summary}\n")
            file.write("\n" + "-" * 50 + "\n")  # Separator line for better readability

        print("Processing completed. Overall summary written to", output_file)

    except Exception as e:
        print(f"Error occurred while processing the video: {e}")


if __name__ == "__main__":
    video_path = r"D:\Nugzar\repo\SkinTypeRecognizer\Assets\oil-vid3.mp4"  # Path to the video file
    output_file = 'video_predictions_summary.txt'

    process_video(video_path, output_file)
