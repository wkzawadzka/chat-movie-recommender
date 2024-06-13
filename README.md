## About

A movie recommender based on 4 different methods: BERT, TF-IDF, T5 & Word2Vec. Let's you write a query and finds most suitable movies based on movies overviews from [TheMovieDatabase](https://www.themoviedb.org/).

## Table of Contents

- [How to Run](#how-to-run)
- [Demo](#demo)
- [Testing](#testing)
- [Methods](#methods)
- [App architecture](#achitecture)
- [Takeaways](#takeaways)

## How to Run

### (0) Download movie data

Go to [Kaggle's MoveLens 1M Dataset](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset) and download. Extract in `backend-djago/data` folder.

"These files contain 1,000,209 anonymous ratings of approximately 3,900 movies
made by 6,040 MovieLens users who joined MovieLens in 2000."

There is also 20M MovieLens dataset, but 1M is enough for our use.

Next, go to [Kaggle's The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and do the same.

### (1) Set up API

Go to [TheMovieDatabase](https://www.themoviedb.org/): signup, create API key and put it in `config.ini`. Put config.ini both in backend/data & frontend/public.

### (2) Enrich movies data by overviews (plots) from API

Run overview retrieval: `python fetch_overviews.py` in `backend-django/chat`. It will save movies.csv in the data folder.

### (3) Take care of CORS issues

Download this Firefox add-on:

https://addons.mozilla.org/en-US/firefox/addon/access-control-allow-origin/

### (4) Run the app

Run `npm start` in the frontend folder and `python manage.py runserver` in backend.

Takeaway: first run can take quite long, models are initializing etc.

## Demo

https://github.com/wkzawadzka/chat-movie-recommender/assets/49953771/18258799-13c4-439b-856b-38e59c315794

## Testing

Run `python manage.py test chat/unittests`.

```
Ran 4 tests in 81.585s

OK
```

## Methods

All the methods used cosine similarity and as for the data, overviews, so short desciption of the movies have been used.

Our 4 methods include:

1. **BERT**

   We used BERT (Bidirectional Encoder Representations from Transformers) to generate contextualized word embeddings for our overviews and the query.

2. **T5**

   T5 (Text-To-Text Transfer Transformer) is employed to convert descriptions into a form (embeddings) where their similarities can be measured effectively using cosine similarity.

3. **TF-IDF**

   Term Frequency Inverse Document Frequency, so multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. TF: Measures, how often a term appears
   • Assuming that important terms appear more often
   • Normalization has to be done in order to take document length into account
   IDF: Reduce the weight of terms that appear in all documents.

4. **Word2Vec**

   Word2Vec is a shallow, two-layer neural network model used to produce word embeddings. Word2Vec captures semantic relationships between words, allowing it to produce vectors where words with similar meanings are close to each other in the vector space. By applying Word2Vec to movie descriptions, we can compute cosine similarity between these vector representations to assess how similar the descriptions are.

## Architecture

The architecture we chose for similar movies recommendations consists in two components. First, we made a backend with the framework Django, to handle data processing. Then, we built an interactive frontend with ReactJS, to provide a user-friendly interface for movie similarity analysis. Results in [Demo](#demo).

## Takeaways

Our overviews dataset has only 3771 movies in it, so the recommendations are not the best, when one is looking for something more specific and rare. It would be better to have much larger dataset, and maybe also play with removing too niche movies as not to recommend to obsure ones, as usually the popularity of recommended movies is a good first and straight-away metric to look at.

---

Project by Eliza Czaplicka, Weronika Zawadzka & Filip Firkowski.
