import logging
import os
from pathlib import Path
from joblib import dump

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
root_folder = str(Path(__file__).resolve().parent.parent)
model_name = "ml2_model.joblib"


def check_and_get_df(csv_file_path: str) -> (bool, pd.DataFrame):
    """
    Function that checks a given path and loads training data, if the path is ok.
    """
    # check if given path exists
    if not os.path.exists(csv_file_path):
        logger.error("File not found, impossible to proceed with the training.")
        return False, pd.DataFrame()

    df = pd.read_csv(csv_file_path, sep=",")
    # check if all need columns are there
    if not pd.Series(["Title", "Description", "Genre"]).isin(df.columns).all():
        logger.error("DataFrame not in the desired format, impossible to proceed with the training.")
        return False, pd.DataFrame()
    else:
        return True, df


def compute_unique_genres(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Function that computes a new column 'UniqueGenre', which associates only one genre to a single movie.
    """
    df_train["GenreList"] = df_train["Genre"].str.split(",")
    genres_per_priority = [
        "Animation", "Biography", "Horror", "Fantasy", "Sci-Fi", "Adventure", "Action", "Crime", "Thriller", "Comedy",
        "Drama", "Mystery", "War", "Sport", "Music", "Romance", "Musical", "Western", "History", "Family"
    ]
    dict_priority = {k: i for i, k in enumerate(genres_per_priority)}
    # keep only one genre per movie according to the list of priorities
    df_train["UniqueGenre"] = df_train["GenreList"].apply(lambda x: x[np.argmin([dict_priority[g] for g in x])])
    # further map two genres, to reduce the number of final genres
    genre_mapping = {g: g for g in df_train["UniqueGenre"].unique()}
    genre_mapping["Crime"] = "Thriller"
    genre_mapping["Fantasy"] = "Adventure"
    df_train["UniqueGenre"] = df_train["UniqueGenre"].map(genre_mapping)
    logger.info(df_train["UniqueGenre"].unique())
    return df_train


def train_from_file(csv_file_path: str):
    """
    Function that do all the training chain from a file containing the training set.
    A Pickle file corresponding to the trained model is eventually saved.
    """
    logger.info("Loading and processing training data.")
    flag, df_train = check_and_get_df(csv_file_path)
    if flag:
        df_train = compute_unique_genres(df_train)
        # get training text data (concatenation of title and description) and labels (the unique genres)
        x_train = (df_train["Title"] + " " + df_train["Description"]).tolist()
        y_train = df_train["UniqueGenre"].tolist()
        # define the model
        model = Pipeline([
            ('vect', CountVectorizer(analyzer="word", strip_accents="unicode", lowercase=True,
                                     max_df=.4, min_df=2, max_features=1500)),
            ('clf', LogisticRegression(solver="lbfgs", multi_class="multinomial", random_state=2020))
        ])
        # fit the model on training data
        logger.info("Fitting the model.")
        model.fit(x_train, y_train)
        # save the trained model
        save_path = os.path.join(root_folder, "resources", model_name)
        dump(model, save_path)
        logger.info(f"Model trained and saved to <{save_path}>.")
