from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render

from chat.utility import *
# from core.core import settings
from .models import *
from core.settings import USER_DATA
from core.settings import MOVIES_DATA
from core.settings import MOVIE_METADATA
from core.settings import CREDITS_DATA
from core.settings import RATINGS_DATA
from core.settings import PLOTS_DATA
from core.settings import BERT_SIM_MATRIX
import time
import json

import numpy as np

from .models import Strategy


users = load_data(USER_DATA)
movies = load_data(MOVIES_DATA)
credits = pd.read_csv(CREDITS_DATA)
plots = pd.read_csv(PLOTS_DATA)
movies_metadata = pd.read_csv(MOVIE_METADATA, low_memory=False)
movies.columns = ["movieID", "title", "genre"]
movies["genre"] = movies["genre"].str.split("|")
users.columns = ["userID", "gender", "age", "occupation", "zip-code"]
ratings = load_data(RATINGS_DATA)
ratings.columns = ["userID", "movieID", "rating", "timestamp"]
bert_sim_matrix = np.load(BERT_SIM_MATRIX)

movies, ratings = limit_movies(
    movies, ratings, threshold=300)
movies = movies.reset_index()


def get_strategy(request, movie):
    print("time start")
    start_time = time.time()  # Start measuring the execution time
    # plots = load_data(PLOTS_DATA, sep=",", header=0)
    # Retrieve all MovieLens users from the database
    index_json = 0
    matching_movies = movies[movies['title'] == movie]

    if len(matching_movies) > 0:
        movie_id = matching_movies.iloc[0]['movieID']

    if request.method == "GET":

        # Get recommendations for the user
        recommendationSVD = Strategy.recommendSVD(ratings, movies, movie_id)
        pearson_strategy = Strategy.recommend_pearson_correlation(
            movies, ratings, movie_id)
        recommendationDice = Strategy.recommend_Dice_strategy(movies, movie_id)
        recommendationTF_IDF = Strategy.recommend_TF_IDF(
            movies, plots, movie_id)
        recommendationBERT = Strategy.recommend_BERT(
            movies, movie_id, bert_sim_matrix)
        result = {}
        for movieID in recommendationBERT[:5]:
            movie_title = movies[movies['movieID']
                                 == movieID]['title'].values[0]

            actors, overview = get_overview_actor(
                movieID, credits, movies_metadata)

            # Create a dictionary with relevant movie details
            new_series = {
                "movieID": movieID,
                "title": movie_title,
                "genre": movies[movies['movieID'] == movieID]['genre'].values[0],
                "actors": actors,
                "overview": overview,
            }

            result[int(index_json)] = new_series
            index_json += 1

        for i, movieID in enumerate(recommendationDice[:5]):
            movie_title = movies[movies['movieID']
                                 == movieID]['title'].values[0]

            actors, overview = get_overview_actor(
                movieID, credits, movies_metadata)

            # Create a dictionary with relevant movie details
            new_series = {
                "movieID": movieID,
                "title": movie_title,
                "genre": movies[movies['movieID'] == movieID]['genre'].values[0],
                "actors": actors,
                "overview": overview,
            }

            result[int(index_json)] = new_series
            index_json += 1

        for i, movieID in enumerate(recommendationTF_IDF[:5]):

            movie_title = movies[movies['movieID']
                                 == movieID]['title'].values[0]
            actors, overview = get_overview_actor(
                movieID, credits, movies_metadata)

            # Create a dictionary with relevant movie details
            new_series = {
                "movieID": movieID,
                "title": movie_title,
                "genre": movies[movies['movieID'] == movieID]['genre'].values[0],
                "actors": actors,
                "overview": overview,
            }

            result[int(index_json)] = new_series
            index_json += 1
        for index, row in recommendationSVD.head(5).iterrows():
            movieID = row['movieID']
            movie_title = row['title']

            actors, overview = get_overview_actor(
                movieID, credits, movies_metadata)

            # Create a dictionary with relevant movie details
            new_series = {
                "movieID": movieID,
                "title": movie_title,
                "genre": movies[movies['movieID'] == movieID]['genre'].values[0],
                "actors": actors,
                "overview": overview,
            }
            result[int(index_json)] = new_series
            index_json += 1

        for i, movieID in enumerate(pearson_strategy[:5]):

            movie_title = movies[movies['movieID']
                                 == movieID]['title'].values[0]

            actors, overview = get_overview_actor(
                movieID, credits, movies_metadata)

            # Create a dictionary with relevant movie details
            new_series = {
                "movieID": movieID,
                "title": movie_title,
                "genre": movies[movies['movieID'] == movieID]['genre'].values[0],
                "actors": actors,
                "overview": overview,
            }

            result[int(index_json)] = new_series
            index_json += 1

        execution_time = time.time() - start_time

        print(f"Execution time: {execution_time} seconds")

        return JsonResponse({"0": result})

    # If the user ID doesn't exist or the request method is not GET, return False as a JSON response
    return JsonResponse({"0": False})
