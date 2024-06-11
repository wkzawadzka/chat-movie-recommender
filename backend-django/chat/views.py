from django.http import HttpResponse, HttpResponseRedirect, JsonResponse  # type: ignore
from django.shortcuts import render  # type: ignore

from chat.utils import *
from .models import BERT
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# ────────────────────────────────────────────────────────────────────────
#                            GET DATA PATHS
# ────────────────────────────────────────────────────────────────────────
# from core.settings import USER_DATA
from core.settings import MOVIES_DATA
from core.settings import MOVIE_METADATA
from core.settings import CREDITS_DATA
from core.settings import PLOTS_DATA

# ────────────────────────────────────────────────────────────────────────
#                            IMPORT DATA
# ────────────────────────────────────────────────────────────────────────
# users
# users = pd.read_csv(USER_DATA, sep="::", header=None, engine="python", encoding="ISO-8859-1")
# users.columns = ["userID", "gender", "age", "occupation", "zip-code"]

# movies
movies = pd.read_csv(MOVIES_DATA, sep="::", header=None,
                     engine="python", encoding="ISO-8859-1")
movies.columns = ["movieID", "title", "genre"]  # type: ignore
movies["genre"] = movies["genre"].str.split("|")

# credits
credits = pd.read_csv(CREDITS_DATA)

# overviews
plots = pd.read_csv(PLOTS_DATA)

# meta
movies_metadata = pd.read_csv(MOVIE_METADATA, low_memory=False)

# ratings
# ratings = pd.read_csv(RATINGS_DATA, sep="::", header=None, engine="python", encoding="ISO-8859-1")
# ratings.columns = ["userID", "movieID", "rating", "timestamp"]

# ────────────────────────────────────────────────────────────────────────
#                            SET UP MODELS
# ────────────────────────────────────────────────────────────────────────
bert = BERT(plots)

# ────────────────────────────────────────────────────────────────────────
#                                  GET
# ────────────────────────────────────────────────────────────────────────


def recommend(request: Any, query: str) -> JsonResponse:
    if request.method != "GET":
        return JsonResponse({"0": False})

    recommendationBERT = bert.recommend(query)
    result: Dict[int, Dict[str, Any]] = {}
    id: int = 0
    for movieID in recommendationBERT:
        # movieID = plots.loc[idx, 'movieID']
        # print(f"idx: {idx} (= 257?), movieID: {movieID} (==260?)")
        info = plots[plots['movieID'] == movieID]
        title = info['title'].values[0]
        plot = info['overview'].values[0]
        actors = get_actors(
            movieID, credits)

        res: Dict[str, Any] = {
            "movieID": str(movieID),
            "title": title,
            "genre": movies[movies['movieID'] == movieID]['genre'].values[0],
            "actors": actors,
            "overview": plot,
        }

        result[id] = res
        id += 1

    return JsonResponse({"0": result})
