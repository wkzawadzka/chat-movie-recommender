from django.http import JsonResponse  # type: ignore
from chat.utils import get_actors
from .models import BERT, T5Predictor, TFIDF, Word2VecModel
import pandas as pd
from typing import Dict, Any

# ────────────────────────────────────────────────────────────────────────
#                            GET DATA PATHS
# ────────────────────────────────────────────────────────────────────────
from core.settings import MOVIES_DATA
from core.settings import MOVIE_METADATA
from core.settings import CREDITS_DATA
from core.settings import PLOTS_DATA

# ────────────────────────────────────────────────────────────────────────
#                            IMPORT DATA
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
#                            SET UP MODELS
# ────────────────────────────────────────────────────────────────────────
bert = BERT(plots)
t5 = T5Predictor(plots)
tfidf = TFIDF(plots)
word2vec = Word2VecModel(plots)

# ────────────────────────────────────────────────────────────────────────
#                                  GET
# ────────────────────────────────────────────────────────────────────────


def recommend(request: Any, query: str) -> JsonResponse:
    if request.method != "GET":
        return JsonResponse({"0": False})

    recommendationBERT = bert.recommend(query)
    recommendationT5 = t5.recommend(query)
    recommendationTFIDF = tfidf.recommend(query)
    recommendationWord2Vec = word2vec.recommend(query)

    result: Dict[int, Dict[str, Any]] = {}
    id: int = 0
    for movieID in recommendationT5 + recommendationTFIDF + recommendationWord2Vec + recommendationBERT:
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
