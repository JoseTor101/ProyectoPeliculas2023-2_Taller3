from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Movie, Review
from .forms import ReviewForm
from movie.models import Movie
import os
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv, find_dotenv


def home(request):
    searchTerm = request.GET.get('searchMovie')
    if searchTerm: 
       movies = Movie.objects.filter(title__icontains=searchTerm) 
    else: 
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm':searchTerm, 'movies': movies})


def about(request):
    return render(request, 'about.html')


def detail(request, movie_id):
    movie = get_object_or_404(Movie,pk=movie_id)
    reviews = Review.objects.filter(movie = movie)
    return render(request, 'detail.html',{'movie':movie, 'reviews': reviews})

def recommendations(request):
    searchTerm = request.GET.get('searchRecommendation')
    films = []
    if searchTerm: 
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['api_key']
        items = Movie.objects.all()

        req = searchTerm
        emb_req = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)

        top_3_indices = np.argsort(sim)[-3:][::-1]
        
        print("Las 3 películas más similares son:")
        for idx in top_3_indices:
                movie_title = items[int(idx)]
                movie = Movie.objects.filter(title = movie_title)
                films.append(movie)
                similarity_score = sim[(idx)]
        print(films, "💗")
    else: 
        print("No hay resultados")
    return render(request, 'recommendations.html',{'searchRecommendation':searchTerm, 'films': films })

@login_required
def createreview(request, movie_id):
    movie = get_object_or_404(Movie,pk=movie_id)
    if request.method == 'GET':
        return render(request, 'createreview.html',{'form':ReviewForm(), 'movie': movie})
    else:
        try:
            form = ReviewForm(request.POST)
            newReview = form.save(commit=False)
            newReview.user = request.user
            newReview.movie = movie
            newReview.save()
            return redirect('detail', newReview.movie.id)
        except ValueError:
            return render(request, 'createreview.html',{'form':ReviewForm(),'error':'bad data passed in'})

@login_required       
def updatereview(request, review_id):
    review = get_object_or_404(Review,pk=review_id,user=request.user)
    if request.method =='GET':
        form = ReviewForm(instance=review)
        return render(request, 'updatereview.html',{'review': review,'form':form})
    else:
        try:
            form = ReviewForm(request.POST, instance=review)
            form.save()
            return redirect('detail', review.movie.id)
        except ValueError:
            return render(request, 'updatereview.html',{'review': review,'form':form,'error':'Bad data in form'})
        
@login_required
def deletereview(request, review_id):
    review = get_object_or_404(Review, pk=review_id, user=request.user)
    review.delete()
    return redirect('detail', review.movie.id)