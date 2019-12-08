import json
import logging
import os
from pathlib import Path

from joblib import load
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
root_folder = str(Path(__file__).resolve().parent.parent)
model_name = "ml2_model.joblib"


def load_model() -> Pipeline:
    """
    Function that loads the serialized model generated with the training script.
    """
    model_path = os.path.join(root_folder, "resources", model_name)
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        logger.error("Model file not found, impossible to proceed with the prediction.")
        model = None
    return model


def predict_genre(title: str, description: str) -> str:
    """
    Function that do the prediction of the genre of a movie from its title and description.
    It return a JSON-formatted string with the predicted genre as extra information, if no problem occurred.
    """
    response_dict = {"title": title, "description": description}
    logger.info("Loading the model.")
    model = load_model()
    if model:
        logger.info("Computing the prediction.")
        x_test = [title + " " + description]
        predicted_genre = model.predict(x_test)[0]
    else:
        predicted_genre = ""
    response_dict["genre"] = predicted_genre

    return json.dumps(response_dict, indent=4)