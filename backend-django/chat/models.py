from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chat.utility import *
from functools import cached_property, cache
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from django.db import models

# import warnings
# warnings.filterwarnings("ignore")

# ''' desciption main 5 strategies '''


class Strategy(models.Model):

    def recommendSVD(ratings, movies, query: int, k: int = 10):
        '''
        @inputs
            query: movieID of movie of interest
        @outputs
            recommendation: list of top k movies with highest similarity score
        '''

        # Extract the relevant columns from the movies DataFrame
        movie_data = movies[["movieID", "title", "genre"]].copy()

        # Perform SVD on the ratings DataFrame
        svd = TruncatedSVD(n_components=50, random_state=42)
        ratings_matrix = ratings.pivot(
            index="userID", columns="movieID", values="rating").fillna(0)
        svd.fit(ratings_matrix.T)  # Transpose the ratings_matrix
        # Get the index of the target movie
        target_movie_index = movie_data[movie_data["movieID"]
                                        == query].index[0]

        # Calculate the latent factors for movies
        movie_latent_factors = svd.transform(
            ratings_matrix.T)  # Transpose the ratings_matrix

        # Get the genres of the target movie
        target_genres = movie_data[movie_data["movieID"]
                                   == query]["genre"].iloc[0]

        # Calculate the similarity scores using a combination of latent factors and genre information
        similarity_scores = []
        for i, (latent_factors, genres) in enumerate(zip(movie_latent_factors, movie_data["genre"])):
            latent_factor_similarity = np.dot(
                latent_factors, movie_latent_factors[target_movie_index])
            if len(set(genres).intersection(target_genres)) > 0:
                similarity_score = latent_factor_similarity * \
                    len(set(genres).intersection(target_genres))
            else:
                similarity_score = latent_factor_similarity

            similarity_scores.append(similarity_score)

        # Normalize the similarity scores between -1 and 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_scores = scaler.fit_transform(
            np.array(similarity_scores).reshape(-1, 1))

        # Sort the movies based on similarity scores in descending order
        sorted_indices = np.argsort(normalized_scores, axis=0)[::-1].squeeze()

        # Retrieve the top k similar movies excluding the query movie
        top_indices = sorted_indices[sorted_indices != target_movie_index][:k]

        # Get the movieIDs of the recommended movies
        recommended_movieIDs = movie_data.iloc[top_indices]["movieID"].tolist()
        recommended_movieIDs = getNFirst(movies, recommended_movieIDs, 5)
        # Filter the recommended movies based on the retrieved movieIDs
        recommended_movies = movie_data[movie_data["movieID"].isin(
            recommended_movieIDs)].copy()

        return recommended_movies

    def recommend_pearson_correlation(movies, ratings, query: int, k=10):
        movieIDs = set(movies['movieID'])
        movieIDs.remove(query)

        similarities = {}
        vectorRef = ratings[ratings['movieID'] == query]

        for movieID in movieIDs:
            sim = pearson_correlation(
                vectorRef, ratings[ratings['movieID'] == movieID])
            similarities[movieID] = sim
        result = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        result = [elt[0] for elt in result]
        return getNFirst(movies, result, 5)

    def recommend_Dice_strategy(movies, query: int, k: int = 10):
        '''         
        @inputs
            query: movieID of movie of intrest 
        @outputs
            recommendation: list of top k movies with highest similarity score
        '''

        searched_movie_genres = movies.loc[movies['movieID']
                                           == query, 'genre'].values[0]
        # Calculate the similarity of each movie to the searched movie
        similarities = {}
        for index, row in movies.iterrows():
            if row['movieID'] != query:
                similarity = dice_coefficient(
                    searched_movie_genres, row['genre'])
                similarities[row['movieID']] = similarity

        # Sort the movies in descending order of similarity
        sorted_movies = sorted(similarities.items(),
                               key=lambda x: x[1], reverse=True)

        # # Print the top recommended movies
        # recommended_movies = []
        # for movie_id, similarity_score in sorted_movies[:k]:
        #     movie_title = self.movies.loc[self.movies['movieID'] == movie_id, 'title'].values[0]
        #     genres = self.movies.loc[self.movies['movieID'] == movie_id, 'genre'].values[0]
        #     recommended_movies.append((movie_title, genres, similarity_score))

        result = [elt[0] for elt in sorted_movies]
        return getNFirst(movies, result, 5)

    def recommend_TF_IDF(movies, plots, query: int, k: int = 10):
        plot = (plots[plots['movieID']
                      == query]['overview']).values[0]
        plots = compute_tokens_plot(plots)
        wordFrequencies, vocabulary, lenVocabulary = calculateFrequenciesDocs(
            plots, {}, 0)
        tfIdfDic = compute_vectors(plots, vocabulary, wordFrequencies)
        similarMovies = findNearestMovies(
            movies, query, plot, tfIdfDic, vocabulary, plots, k)

        return getNFirst(movies, similarMovies, 5)

    def recommend_BERT(movies, query: int, similarity_matrix, k: int = 10):
        movie_ids = movies['movieID'].values.tolist()
        query_idx = movie_ids.index(query)
        similarities = pd.Series(
            similarity_matrix[query_idx]).sort_values(ascending=False)
        # ommiting query itself in the result
        top_k_indicies = list(similarities.iloc[1:k+1].index)
        result = [movie_ids[i] for i in top_k_indicies]

        return getNFirst(movies, result, 5)
