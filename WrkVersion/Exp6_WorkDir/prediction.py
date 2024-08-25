import os
import inference
import cv2
from Assets import dry

# Set API key
os.environ["ROBOFLOW_API_KEY"] = "8Qs7F54MSyWFOmMq8Yva"


def predict_image(image):
    model = inference.get_roboflow_model("skin-type-tgow5/1")

    if image is None:
        raise ValueError(f"Unable to open image file: {image_path}")

    # Convert image to RGB (if necessary)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use the image in the inference call
    results = model.infer(image=image_rgb)
    predictions = results[0].predictions
    return {label: prediction.confidence for label, prediction in predictions.items()}
