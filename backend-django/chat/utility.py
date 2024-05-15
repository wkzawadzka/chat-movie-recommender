import sys
import os
import requests
import pandas as pd
import numpy as np
from PIL import Image
import re
from io import BytesIO

import json
import ast
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from collections import Counter


def tokenize(plot):
    '''
        Converts a string to a list of tokens, remove the stopwords and lemmatize all the tokens
    '''
    lemmatizer = WordNetLemmatizer()
    plot = word_tokenize(plot)
    rep = []
    stopwordsList = stopwords.words('english')
    # we also want to remove the number of the movie episode
    stopwordsList += ['The', 'Episode', 'Part', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
                      'IX', 'X', 'XI', 'XII', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    for token in plot:
        if token not in stopwordsList:
            rep.append(lemmatizer.lemmatize(token.lower()))
    return rep


def compute_tokens_plot(plots):
    '''
        create a dictionary with key : movieID and value : tokens list of the movie title (thanks to movies dataset)
    '''
    result = {}
    for index, row in plots.iterrows():
        plot = str(row['overview'])
        plot = tokenize(plot)
        # because of stopwords removal, we can have the same list of tokens for different movies (example : Toy Story and Toy Story 2 will both return ['Toy', 'Story']
        # the next line enables to have only one time the same list of tokens in result
        if plot not in result.values():
            result[row['movieID']] = plot
    return result


def calculateFrequenciesDocs(plots, vocabulary, lenVocabulary, IdfWeight=1):
    '''create a dictionary with :
    -key : a word encountered in the movie plot
    -value : dictionary with key : movieID, value : TF-IDF for this movie and this word

    Vocabulary is built inside this function. At the end, it contains all the lemmatized tokens (not stopwords) encountered in all the movie titles.
    It's a dictionary with key : id of the word and value : word

    plots is the result of compute_tokens_title
    '''
    wordFrequencies = {}

    # TF
    for movieId, token_list in plots.items():
        nbrOfWords = len(token_list)
        wordCounts = Counter(token_list)

        for word in wordCounts.elements():
            if word not in vocabulary.values():
                lenVocabulary += 1
                vocabulary[lenVocabulary] = word
                wordFrequencies[word] = {
                    movieId: wordCounts[word] / nbrOfWords}
            else:
                wordFrequencies[word][movieId] = wordCounts[word] / nbrOfWords

    # IDF
    for word in wordFrequencies.keys():
        occurence = len(wordFrequencies[word])
        for movie in wordFrequencies[word].keys():
            wordFrequencies[word][movie] *= (
                np.log(len(plots) / occurence + 1)) ** IdfWeight
    return wordFrequencies, vocabulary, lenVocabulary


def compute_vectors(plots, vocabulary, wordFrequencies):
    '''
    computes a dictionary with key : movieId and value : vector which represents this movie
    The vector contains as many values as there are words in the vocabulary. vector[i] is the TF-IDF value of the word whose id is i in vocabulary
    '''
    tfIdfDic = {}
    for movieId in plots.keys():
        vector = []
        for word in vocabulary.values():
            if movieId in wordFrequencies[word]:
                vector.append(wordFrequencies[word][movieId])
            else:
                vector.append(0)
        tfIdfDic[movieId] = vector
    return tfIdfDic


def toVector(plot, vocabulary):
    '''
    converts a plot to a vector
    The vector size is the number of words in the vocabulary.
    For a word whose id is i in the vocabulary, vector[i] = 1 if the word appears in the plot vector[i] = 0 if not
    '''
    plot = tokenize(plot)
    vector = []
    for word in vocabulary.values():
        if word in plot:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def cosine(vector1, vector2):
    '''
    compute the similarity beetween two vectors thanks to a cosine formula (vector1.vector2)/(norm(vector1).norm(vector2)
    '''
    n1 = np.linalg.norm(vector1)
    if n1 == 0:
        return 0
    n2 = np.linalg.norm(vector2)
    if n2 == 0:
        return 0
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def findNearestMovies(movies, movieIdRef, plot, tfIdfDic, vocabulary, plots, k):
    '''
    returns the five nearest movies of movieIdRef using the cosine function and tf.idf vectors
    similarities is a dictionary with key : movieId and value : similarity betwwen the movie movieId and the movie movieIdRef
    '''
    similarities = {}
    vectorRef = toVector(plot, vocabulary)
    tokensList = tokenize(plot)
    # we don't want to recommend a movie with the same tokens as the input one, so we delete the (only) entry of the dictionary whose value is this list of tokens
    keys = list(tfIdfDic.keys())
    for key in keys:
        if tokensList == plots[key]:
            tfIdfDic.pop(key)
    for id, vector in tfIdfDic.items():
        similarities[id] = cosine(vector, vectorRef)
    result = sorted(similarities.items(),
                    key=lambda x: x[1], reverse=True)[:k]
    result = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    result = [elt[0] for elt in result]
    return getNFirst(movies, result, 5)


def dice_coefficient(genre1, genre2):
    genre1_count = Counter(genre1)
    genre2_count = Counter(genre2)
    intersection = len(list((genre1_count & genre2_count).elements()))
    return 2.0 * intersection / (len(genre1) + len(genre2))


def pearson_correlation(ratings_A: pd.DataFrame, ratings_B: pd.DataFrame):
    ''' calculates similarity measure using Pearson crrelation
        returns possible similarity values betweeen -1 and 1 '''

    users_who_rated_both = set(ratings_A['userID']).intersection(
        set(ratings_B['userID']))  # set intersection of movies

    # if they did not rate any same movies, correlation is -1
    if len(users_who_rated_both) == 0:
        return -1

    # otherwise, calculate the value of correlation

    mean_a = sum(ratings_A['rating'])/len(ratings_A)
    mean_b = sum(ratings_B['rating'])/len(ratings_B)

    nominator = sum((ratings_A[ratings_A['userID'].isin(
        users_who_rated_both)]['rating'] - mean_a).values*(ratings_B[ratings_B['userID'].isin(
            users_who_rated_both)]['rating'] - mean_b).values)

    denominator = (sum((ratings_A[ratings_A['userID'].isin(
        users_who_rated_both)]['rating'] - mean_a)**2))**(1/2)*(sum((ratings_B[ratings_B['userID'].isin(
            users_who_rated_both)]['rating'] - mean_b)**2))**(1/2)

    # if the standard deviation of one of the variables is zero, then the denominator is zero and the correlation cannot be computed
    # therefore we chose to output correlation coefficient as 0
    if denominator == 0:
        return 0

    return nominator/denominator


def get_overview_actor(item, credits, movies_metadata):
    d = ast.literal_eval(credits["cast"][item - 1])
    json_str = json.dumps(d, indent=4)
    d = pd.read_json(json_str)
    # Error handling check
    if len(d) > 0:
        actors = d["name"][:3].to_list()
    else:
        actors = []

    # Retrieve director information
    # crew = ast.literal_eval(credits["crew"][item - 1])
    # json_str = json.dumps(crew, indent=4)
    # crew_data = pd.read_json(json_str)
    # director_jobs = crew_data[crew_data["job"].str.contains(r'\bdirector\b', flags=re.IGNORECASE)]["name"].tolist()

    if type(movies_metadata.iloc[item]["overview"]) == type(np.nan):
        overview = "no overview"
    else:
        overview = movies_metadata.iloc[item]["overview"]

    return actors, overview


def load_data(path) -> pd.DataFrame:
    """loading dataset with error handling
    returns loaded dataset
    """

    try:
        df = pd.read_csv(
            path, sep="::", header=None, engine="python", encoding="ISO-8859-1"
        )

    except FileNotFoundError:
        print(f"File {path} could not be found")
        sys.exit()

    except OSError:
        print(f"File {path} could not be read")
        sys.exit()

    return df


def install_and_import(package):
    ''' installs a new package; used for problems with imports '''
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


def get_user_input(movies: pd.DataFrame) -> int:
    ''' accepts an movie ID as an input (on the console) with error handling '''
    allowableIDs = set(movies['movieID'])

    print("Please input movie ID to begin:")
    while True:
        try:
            n = input("$")
            # check if it is an integer, otherwise ValueError
            n = int(n)

            # check if given value is present in the dataset
            if n not in allowableIDs:
                raise ValueError

            break

        except ValueError:
            print(
                "Error. Please provide existing integer value of movie ID. Try again...")

    return n


def limit_movies(movies: pd.DataFrame, ratings: pd.DataFrame, threshold=300):
    ''' returns new movie and ratings dataset without movies with less that {threshold} ratings '''
    toDrop = set()
    for movie in set(movies['movieID']):
        if (len(ratings[ratings['movieID'] == movie]) < threshold):
            toDrop.add(movie)

    movies = movies[~movies['movieID'].isin(toDrop)]
    ratings = ratings[~ratings['movieID'].isin(toDrop)]

    return (movies, ratings)


# def get_movie_poster(title, api_key) -> None:
#     ''' downloads movie poster from movie database '''

#     # search for movie with given title in movie database
#     base_url = "https://api.themoviedb.org/3/search/movie"
#     params = {"api_key": api_key, "query": title}
#     data = requests.get(base_url, params=params).json()

#     # get the poster path & plot overview
#     poster_path = data["results"][0]["poster_path"]

#     # construct the URL to download the image
#     base_url = "https://image.tmdb.org/t/p/"
#     size = "w500"  # adjust this to change the size of the image
#     url = f"{base_url}{size}{poster_path}"

#     # download the image and save it
#     response = requests.get(url, verify=False)
#     img = Image.open(BytesIO(response.content))

#     # create the directory if it doesn't exist
#     folder_path = "data/images/"
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     # save the image to the specified folder
#     file_path = f"{folder_path}{title}.jpg"
#     img.save(file_path)

#     print(f"The image for {title} has been saved to {file_path}")


def get_movie_overview(title, api_key) -> str:
    ''' returns movie's plot overview from movie database'''
    print(f"Looking for a oveerview of movie called: {title}")

    # search for movie with given title in movie database
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title}
    data = requests.get(base_url, params=params).json()

    if len(data['results']) == 0:
        return ""

    # get the plot overview
    overview = data["results"][0]["overview"]
    return overview
    # return transform_overview(overview)


def transform_overview(overview: str) -> str:
    ''' prepares the overview for analysis '''
    overview = overview.lower()  # lowercase
    overview = re.sub('[0-9]', '', overview)  # no numbers
    overview = re.sub('[!@?#$%^&*()"<>/`~+=-_.,“‘:;]', '', overview)

    return overview


def clear_title(title: str) -> str:
    ''' remove the date at the end of the default title name'''
    return title[:-7]


def normalization(title: str) -> str:
    '''normalize a movie title by removing the part linked with the episode, the date in the end...'''
    without_dates = re.sub(r'\(\d+\)', '', title)
    transform1 = re.sub(r'(:\s*)?Episode.*', '', without_dates)
    transform2 = re.sub(r'(:\s*)?Part.*', '', transform1)
    transform3 = re.sub(r'(\d+\s*)?:.*', '', transform2)
    transform4 = re.sub(r'\d+\s*$', '', transform3)
    transform5 = re.sub(r'(\d+\s*)?\(.*\)', '', transform4)
    transform6 = re.sub(
        r'((((X{0,3})(IX|IV|V?I{0,3}))|[V|X]((I{0,3})(IX|IV|V?I{0,3})))+\s*)?:.*', '', transform5)
    transform7 = re.sub(
        r'(((X{0,3})(IX|IV|V?I{0,3}))|[V|X]((I{0,3})(IX|IV|V?I{0,3})))+\s*', '', transform6)
    transform8 = re.sub(
        r'((((X{0,3})(IX|IV|V?I{0,3}))|[V|X]((I{0,3})(IX|IV|V?I{0,3})))+\s*)?\(.*\)', '', transform7)
    main_name = re.sub(r',\s*The', '', transform8)
    normalized_name = main_name.lower().strip()
    return normalized_name


def getNFirst(movies, movie_list, n: int):
    '''given a list of movieID, return the n first which have different normalized title (so that it doesn't return several times the same movie with different episodes)'''
    result = []
    list_normalized = []
    count = 0
    index = 0
    while count < n:
        movieID = movie_list[index]
        title = movies[movies['movieID'] == movieID]['title'].values[0]
        normalized_title = normalization(title)
        if normalized_title not in list_normalized:
            list_normalized.append(normalized_title)
            result.append(movieID)
            count += 1
        index += 1
    return result
