## Steps

### (0) Download movie data

Go to [Kaggle's MoveLens 1M Dataset](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset) and download. Extract in `backend-djago/data` folder.

"These files contain 1,000,209 anonymous ratings of approximately 3,900 movies
made by 6,040 MovieLens users who joined MovieLens in 2000."

There is also 20M MovieLens dataset, but 1M is enough for our use.

Next, go to [Kaggle's The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and do the same.

### (1) Set up API

Go to [TheMovieDatabase](https://www.themoviedb.org/): signup, create API key and put it in `config.ini`. Put config.ini both in backend/data & frontend/public.

### (2) Get overviews (plots) from API

Run overview retrieval: `python fetch_overviews.py` in `backend-django/chat`.

### (3) Take care of CORS issues

Download this Firefox add-on:

https://addons.mozilla.org/en-US/firefox/addon/access-control-allow-origin/

### (4) Run

Run `npm start` in the frontend folder and `python manage.py runserver` in backend.
