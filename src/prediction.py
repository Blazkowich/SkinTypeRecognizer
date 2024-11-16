import os
import inference
import cv2
from Assets import dry

# set API key directly or from env variables
os.environ["ROBOFLOW_API_KEY"] = "ROBOFLOW API Key"


def predict_image(image_path):
    model = inference.get_roboflow_model("skin-type-tgow5/1")

    # reading the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to open image file: {image_path}")

    # converting image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # using of the image in the inference call
    results = model.infer(image=image_rgb)
    predictions = results[0].predictions
    return {label: prediction.confidence for label, prediction in predictions.items()}


if __name__ == "__main__":
    image_path = r"your\path"

    try:
        predictions = predict_image(image_path)
        print(predictions)
    except Exception as e:
        print(f"Error occurred: {e}")