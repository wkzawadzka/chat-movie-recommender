import configparser
import pandas as pd
import sys
import requests

config = configparser.ConfigParser()
config.read("../config.ini")
settings = config["settings"]
user_info = config["user_info"]


def load_data(path: str, sep="::", header=None) -> pd.DataFrame:
    '''Loads dataset with error handling.'''
    try:
        df = pd.read_csv(path, sep=sep, header=header, engine="python")
    except FileNotFoundError:
        print(f"File {path} could not be found")
        sys.exit()
    except OSError:
        print(f"File {path} could not be read")
        sys.exit()
    return df


def clear_title(title: str) -> str:
    '''Remove the date at the end of the default title name.'''
    return title[:-7]


def limit_movies(movies: pd.DataFrame, ratings: pd.DataFrame, threshold=150):
    '''Returns new movie and ratings dataset without movies with less than {threshold} ratings.'''
    to_drop = set()
    for movie in set(movies['movieID']):
        if len(ratings[ratings['movieID'] == movie]) < threshold:
            to_drop.add(movie)
    movies = movies[~movies['movieID'].isin(to_drop)]
    ratings = ratings[~ratings['movieID'].isin(to_drop)]
    return movies, ratings


def get_movie_overview(title, api_key) -> str:
    '''Returns movie's plot overview from movie database.'''
    print(f"Looking for an overview of the movie called: {title}")

    # Search for movie with the given title in the movie database
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    data = requests.get(base_url, params=params).json()

    if len(data['results']) == 0:
        return ""

    # Get the plot overview
    overview = data["results"][0]["overview"]
    return overview


if __name__ == '__main__':
    API_KEY = user_info["api_key"]

    # Load movies dataset
    movies = load_data("./data/movies.dat")
    movies.columns = ['movieID', 'title', 'genre']
    movies = movies.drop(['genre'], axis=1)

    ''' Limiting number of movies: we will drop those with less than 150 reviews
        as to not recommend too obscure ones '''
    ratings = load_data("./data/ratings.dat")
    ratings.columns = ['userID', 'movieID', 'rating', 'timestamp']
    movies, ratings = limit_movies(
        movies, ratings, int(settings["drop_threshold"]))

    # Append overview to each movie
    overviews = {title: get_movie_overview(
        clear_title(title), API_KEY) for title in set(movies['title'])}
    movies['overview'] = movies['title'].map(overviews)

    # Save to a file
    movies.to_csv("./data/plots.csv", index=False)
