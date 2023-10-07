import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# movies["year"] = movies.title.str.extract('(\(\d\d\d\d\))', expand=False)
# movies["year"] = movies.year.str.extract('(\d\d\d\d)', expand=False)

# movies["title"] = movies.title.str.replace('(\(\d\d\d\d\))', '')
# movies["title"] = movies["title"].apply(lambda x: x.strip())

# print("Movies: \n", movies.head())
# print("Movies: \n", movies)

movies["genres"] = movies.genres.str.split("|")

movies_co = movies.copy()

for index, row in movies.iterrows():
    for genre in row["genres"]:
        movies_co.at[index, genre] = 1

movies_co = movies_co.fillna(0)

# print("\n Encoded movies: \n", movies_co)

ratings = ratings.drop("timestamp", axis=1)
# print("\n Rating: \n", ratings.head())

user_en = [
    {"title": "Ace Ventura: When Nature Calls (1995)", "rating": 2},
    {"title": "Titanic (1997)", "rating": 4},
    {"title": "Jumanji (1995)", "rating": 3},
    {"title": "Pulp Fiction (1994)", "rating": 4},
    {"title": "Don't Breathe (2016)", "rating": 4.5},
    {"title": "Madagascar (2005)", "rating": 4},
    {"title": "Minions (2015)", "rating": 1},
    {"title": "Into the Wild (2007)", "rating": 4.5},
    {"title": "Hush (1998)", "rating": 5},
]

movies_input = pd.DataFrame(user_en)
# # print("\n User Movies: \n", movies_input)

print("\n Movies: \n", movies_co)
print("\n User Movies: \n", movies_input)

Id = movies[movies["title"].isin(movies_input["title"].tolist())]
movies_input = pd.merge(Id, movies_input)

print("\n Movies Input: \n", movies_input)

movies_input = movies_input.drop("genres", axis=1)  # .drop("year", axis=1)

movies_user = movies_co[movies_co["movieId"].isin(
    movies_input["movieId"].tolist())]

print(f"\n Encoded User Movies: \n {movies_user}")

movies_user = movies_user.reset_index(drop=True)
genres_table = movies_user.drop("movieId", axis=1).drop(
    "title", axis=1).drop("genres", axis=1)  # .drop("year", axis=1)

print(f"\n Genres Table: \n {genres_table}")

user_profile = genres_table.transpose().dot(movies_input["rating"])
print(f"\n User category selected: \n {user_profile}")

genres = movies_co.set_index(movies_co["movieId"])

genres = genres.drop("movieId", axis=1).drop(
    "title", axis=1).drop("genres", axis=1)  # .drop("year", axis=1)

print(f"\n Genres: \n {genres.head()}")
genres.shape

recom = ((genres * user_profile).sum(axis=1)) / user_profile.sum()
print(f"\n Recommended: \n {recom.head()}")

recom = recom.sort_values(ascending=False)
print(f"\n Recommended sorted: \n {recom}")

result = movies.loc[movies["movieId"].isin(recom.head(20).keys())]
nresult = result[["title", "genres"]]

print(f"\n 20 Top list Recommended: \n {nresult}")
