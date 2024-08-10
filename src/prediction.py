import os
import inference

# Set API key
os.environ["ROBOFLOW_API_KEY"] = "8Qs7F54MSyWFOmMq8Yva"

def predict_image(image_path):
    model = inference.get_roboflow_model("skin-type-tgow5/1")
    results = model.infer(image=image_path)
    predictions = results[0].predictions
    return {label: prediction.confidence for label, prediction in predictions.items()}
