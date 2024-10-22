# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

# import sys

# # Increase recursion limit
# sys.setrecursionlimit(10000)  # Set to a suitable value based on your requirements


# #Load the dataset
# movies = pd.read_csv('movie.csv')
# ratings = pd.read_csv('rating.csv')    

# #Merge movies and ratings data
# movie_ratings = pd.merge(movies, ratings, on = 'movieId')

# #Create a pivot table of user ratings
# ratings_matrix = movie_ratings.pivot_table(index='userId', columns='title', values = 'rating')

# #Fill NaN Values with 0
# ratings_matrix = ratings_matrix.fillna(0)

# #Calculate cosine similarity between movies
# movie_similarity = cosine_similarity(ratings_matrix.T)

# #Convert the movie similarity matrix into a DataFrame
# movie_similarity_df = pd.DataFrame(movie_similarity, index = ratings_matrix.columns, columns=ratings_matrix.columns)

# def recommend_movies(movie_title, top_n=10):

#     #Get the similarity scores for the given movie
#     similarity_series = movie_similarity_df[movie_title]

#     #Sort by similarity score in descending order
#     sorted_similarity = similarity_series.sort_values(ascending = False)

#     #Get top N most similar movies
#     top_similar_movies = sorted_similarity.drop(movie_title).head(top_n)

#     return top_similar_movies

# def main():
#     movie_title = "Fast X"

#     recommended_movies = recommend_movies(movie_title)  # Change the variable name here

#     print(f"Recommended movies similar to '{movie_title}': ")  # Remove .format(movie_title)
#     print(recommended_movies)


# if __name__=="__main__":
#     main()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display


#downloading the data from https://files.grouplens.org/datasets/movielens/ml-25m.zip
movies = pd.read_csv("Level3_Programs/movies.csv")

movies.head()

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

tdidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tdidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names='value')


display(movie_input, movie_list)
