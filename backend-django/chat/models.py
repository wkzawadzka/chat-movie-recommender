from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from transformers import T5Tokenizer, T5EncoderModel
import os
from tqdm import tqdm
import torch
from typing import List
from nltk.corpus import stopwords  # type: ignore
from transformers import DistilBertTokenizer, DistilBertModel  # type: ignore
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np
from django.db import models  # type: ignore
import spacy  # type: ignore
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))


class BERT(models.Model):
    '''
    Movie recommendation strategy based on BERT model - High-performance semantic similarity.
    Works by finding similarities between movies' overviews:
        (1) creating tokens out of each overview
        (2) sending tokenized overviews through BERT model
        (3) choice of recommendation according to cosine similarity score between model outputs
    '''  # noqa: E501

    def __init__(self,
                 movies: pd.DataFrame,
                 cache_dir: str = './data/') -> None:

        # ensure cache dir exists
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", cache_dir=self.cache_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased', cache_dir=self.cache_dir)
        if self.tokenizer.pad_token is None:
            # add tokenizer
            self.tokenizer.add_special_tokens(
                {'pad_token': 'EOS'})  # end of sentence token
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.df = movies.dropna()
        self.overviews = [self.preprocess_string(
            str(a)) for a in self.df['overview'].values.tolist()]
        print(f"Dataset size: {len(self.overviews)}")

        print("Getting bert embeddings...")
        embeddings_path = os.path.join(
            self.cache_dir, 'overviews_embeddings.pt')
        if not os.path.exists(embeddings_path):
            self.overviews_embeddings = self.model_outputs(self.overviews)
            torch.save(self.overviews_embeddings, embeddings_path)
        else:
            self.overviews_embeddings = torch.load(embeddings_path)
        print("Done")

    def preprocess_string(self, text: str):
        doc = nlp(text)
        cleaned_text = ' '.join([token.lemma_
                                 for token in doc
                                 if token.text.lower()
                                 not in stop_words and token.is_alpha])

        return cleaned_text

    def model_outputs(self, items: List[str], batch_size: int = 64):
        all_outputs = []
        num_batches = len(items) // batch_size + (len(items) % batch_size != 0)

        for i in tqdm(range(0, len(items), batch_size),
                      desc="Progress",
                      total=num_batches):
            batch_items = items[i:i+batch_size]
            inputs = self.tokenizer(batch_items, add_special_tokens=True,
                                    padding=True, max_length=100,
                                    truncation=True, return_tensors="pt")

            with torch.no_grad():
                # [batch, maxlen, hidden_state]
                # -> using only [batch, hidden_state]
                outputs = self.model(
                    **inputs).last_hidden_state[:, 0, :].numpy()
                all_outputs.append(outputs)

        return np.concatenate(all_outputs, axis=0)

    def recommend(self, query: str, k: int = 5):
        '''
        @inputs
            query: string, description of movie you seek for
        @outputs
            result: list of top k movies with highest similarity score
        '''  # noqa: E501
        query_embedding = self.model_outputs([self.preprocess_string(query)])
        sim = cosine_similarity(query_embedding, self.overviews_embeddings)[0]

        movie_ids = self.df['movieID'].values.tolist()
        top_k_indicies = sim.argsort()[-k:][::-1].tolist()

        result = [movie_ids[i] for i in top_k_indicies]
        print(f"Movie IDS: {result}")
        return result


class T5Predictor(models.Model):
    def __init__(self, movies: pd.DataFrame, cache_dir: str = './data/', model_name: str = "t5-small"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.model = T5EncoderModel.from_pretrained(
            model_name, cache_dir=self.cache_dir)

        self.df = movies
        self.overviews = [self.preprocess(str(a))
                          for a in self.df['overview'].values.tolist()]

        print("Getting T5 embeddings...")
        embeddings_path = os.path.join(
            self.cache_dir, 'overviews_embeddings_t5.pt')
        if not os.path.exists(embeddings_path):
            self.overviews_embeddings = self.model_outputs(self.overviews)
            torch.save(self.overviews_embeddings, embeddings_path)
        else:
            self.overviews_embeddings = torch.load(embeddings_path)
        print("Done")

    def preprocess(self, text: str) -> np.ndarray:
        doc = nlp(text)

        cleaned_text = ' '.join([token.lemma_ for token in doc if token.text.lower(
        ) not in stop_words and token.is_alpha])

        return cleaned_text

    def model_outputs(self, items: List[str], batch_size: int = 64):
        all_outputs = []
        num_batches = len(items) // batch_size + (len(items) % batch_size != 0)

        for i in tqdm(range(0, len(items), batch_size), desc="Progress", total=num_batches):
            batch_items = items[i:i+batch_size]
            inputs = self.tokenizer(batch_items, add_special_tokens=True,
                                    padding=True, max_length=100, truncation=True, return_tensors="pt")

            with torch.no_grad():
                # [batch, maxlen, hidden_state] -> using only [batch, hidden_state]
                outputs = self.model(
                    **inputs).last_hidden_state[:, 0, :].numpy()
                all_outputs.append(outputs)
        return np.concatenate(all_outputs, axis=0)

    def recommend(self, input_text: str, top_n: int = 5) -> list:
        output = self.model_outputs([self.preprocess(input_text)])
        # print(output)
        similarities = cosine_similarity(output, self.overviews_embeddings)[0]

        most_similar_indices = similarities.argsort()[-top_n:][::-1]
        movie_ids = self.df['movieID'].values.tolist()

        result = [movie_ids[i] for i in most_similar_indices]
        print(f"Movie IDS: {result}")
        return result


class TFIDF(models.Model):
    def __init__(self, movies: pd.DataFrame, cache_dir: str = './data/'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer()

        self.df = movies
        self.overviews = [self.preprocess(str(a))
                          for a in self.df['overview'].values.tolist()]

        embeddings_path = os.path.join(
            self.cache_dir, 'overviews_embeddings_tfidf.pt')
        if not os.path.exists(embeddings_path):
            self.overviews_embeddings = self.tfidf_vectorizer.fit_transform(
                self.overviews)
            torch.save(self.overviews_embeddings, embeddings_path)
        else:
            self.tfidf_vectorizer.fit(self.overviews)
            self.overviews_embeddings = torch.load(embeddings_path)

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(
            word) for word in tokens if word.isalnum()]
        return ' '.join(filtered_tokens)

    def recommend(self, prompt, top_n=5):
        prompt_processed = self.preprocess(prompt)
        prompt_vector = self.tfidf_vectorizer.transform([prompt_processed])

        similarity_scores = cosine_similarity(
            prompt_vector, self.overviews_embeddings)[0]
        most_similar_indices = similarity_scores.argsort()[-top_n:][::-1]
        movie_ids = self.df['movieID'].values.tolist()

        result = [movie_ids[i] for i in most_similar_indices]
        print(f"Movie IDS: {result}")
        return result


class Word2VecModel(models.Model):
    def __init__(self, movies: pd.DataFrame, cache_dir: str = './data/'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer()

        self.df: pd.DataFrame = movies
        self.overviews = [self.preprocess(str(a))
                          for a in self.df['overview'].values.tolist()]
        self.word2vec_model = Word2Vec(
            self.overviews, vector_size=100, window=5, min_count=1, workers=4)

        embeddings_path = os.path.join(
            self.cache_dir, 'overviews_embeddings_word2vec.pt')
        if not os.path.exists(embeddings_path):
            self.overviews_embeddings = [self.vectorize_sentence(
                a, self.word2vec_model) for a in self.overviews]

            torch.save(self.overviews_embeddings, embeddings_path)
        else:
            self.overviews_embeddings = torch.load(embeddings_path)

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(
            word) for word in tokens if word.isalnum()]
        return ' '.join(filtered_tokens)

    def vectorize_sentence(self, sentence, model):
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    def recommend(self, prompt, top_n=5):
        prompt_processed = self.preprocess(prompt)
        prompt_vector = self.vectorize_sentence(
            prompt_processed, self.word2vec_model)
        similarity_scores = [cosine_similarity([prompt_vector], [overview])[
            0][0] for overview in self.overviews_embeddings]

        # most_similar_indices = similarity_scores.nlargest(top_n).index
        # similarity_scores = cosine_similarity(prompt_vector, self.overviews_embeddings)[0]
        most_similar_indices = np.argsort(similarity_scores)[-top_n:][::-1]
        movie_ids = self.df['movieID'].values.tolist()

        result = [movie_ids[i] for i in most_similar_indices]
        print(f"Movie IDS: {result}")
        return result
