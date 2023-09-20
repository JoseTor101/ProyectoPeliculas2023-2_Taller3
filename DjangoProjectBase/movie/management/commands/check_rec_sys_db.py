from django.core.management.base import BaseCommand
from movie.models import Movie
import os
import numpy as np

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv, find_dotenv

class Command(BaseCommand):
    help = 'Modify path of images'

    def handle(self, *args, **kwargs):

        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['api_key']
        
        items = Movie.objects.all()

        req = "Peliculas de ficcion"
        emb_req = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        #idx = np.argmax(sim)
        #idx = int(idx)
        #print(items[idx].title)
         # Encuentra los índices de las 3 películas más similares
        top_3_indices = np.argsort(sim)[-3:][::-1]
        
        print("Las 3 películas más similares son:")
        for idx in top_3_indices:
                movie_title = items[int(idx)]
                similarity_score = sim[(idx)]
                #print(f"Título: {movie_title}, Similitud (cosine similarity): {similarity_score}")
                print(f"Título: {movie_title}")
        #print("💥", sim, "💥")
