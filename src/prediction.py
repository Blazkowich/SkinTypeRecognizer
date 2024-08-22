import os
import inference
import cv2
from Assets import dry

# Set API key
os.environ["ROBOFLOW_API_KEY"] = "8Qs7F54MSyWFOmMq8Yva"


def predict_image(image_path):
    model = inference.get_roboflow_model("skin-type-tgow5/1")

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to open image file: {image_path}")

    # Convert image to RGB (if necessary)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use the image in the inference call
    results = model.infer(image=image_rgb)
    predictions = results[0].predictions
    return {label: prediction.confidence for label, prediction in predictions.items()}


# Example usage
if __name__ == "__main__":
    image_path = r"D:\Nugzar\repo\SkinTypeRecognizer\Assets\dry\1.jpg"

    try:
        predictions = predict_image(image_path)
        print(predictions)
    except Exception as e:
        print(f"Error occurred: {e}")