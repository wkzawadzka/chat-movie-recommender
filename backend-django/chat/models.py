import os
from tqdm import tqdm
import torch
from typing import List
from nltk.corpus import stopwords
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chat.utils import *
from django.db import models
import spacy
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# import warnings
# warnings.filterwarnings("ignore")


class BERT(models.Model):
    ''' Movie recommendation strategy based on BERT model - High-performance semantic similarity
    Works by finding similarities between movies' overviews:
        (1) creating tokens out of each overviews
        (2) sending tokanized overviews though BERT model
        (3) choice of recommendation according to cosine similarity score between model outputs
    '''

    def __init__(self, movies: pd.DataFrame, cache_dir: str = './data/') -> None:
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
        self.df = movies
        self.overviews = [self.preprocess_string(
            str(a)) for a in self.df['overview'].values.tolist()]

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

    def recommend(self, query: str, k: int = 5):
        '''         
        @inputs
            query: string, description of movie you seek for
        @outputs
            recommendation: list of top k movies with highest similarity score
        '''
        query_embedding = self.model_outputs([self.preprocess_string(query)])
        sim = cosine_similarity(query_embedding, self.overviews_embeddings)[0]

        movie_ids = self.df['movieID'].values.tolist()
        top_k_indicies = sim.argsort()[-k:].tolist()

        return [movie_ids[i] for i in top_k_indicies]
