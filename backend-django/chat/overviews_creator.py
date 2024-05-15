from utility import load_data, get_movie_overview, clear_title, limit_movies
import configparser
config_obj = configparser.ConfigParser()

''' Creates new csv file comining movieID, title and overview of the movie from movie database using their API '''

''' reading config file '''
config_obj.read("./config.ini")
settings = config_obj["settings"]
user_info = config_obj["user_info"]

if __name__ == '__main__':
    # load dataframes
    # https://www.themoviedb.org/ API key
    API_KEY = user_info["api_key"]

    # load movies dataset
    movies = load_data("./data/movies.dat")
    movies.columns = ['movieID', 'title', 'genre']
    movies = movies.drop(['genre'], axis=1)

    # limit movies number
    ratings = load_data("./data/ratings.dat")
    ratings.columns = ['userID', 'movieID', 'rating', 'timestamp']

    # apply the threshold of using movies with more than threshold ratings
    # to avoid to base the research on too obscure (niche) items
    movies, ratings = limit_movies(
        movies, ratings, int(settings["drop_threshold"]))

    # append overview to each movie
    overviews = {title: get_movie_overview(
        clear_title(title), API_KEY) for title in set(movies['title'])}
    movies['overview'] = movies['title'].map(overviews)

    # save to a file
    movies.to_csv("./data/plots.csv", index=False)
