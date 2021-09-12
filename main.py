import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

warnings.filterwarnings('ignore')
columns_names = ["user_id","item_id","rating","timestamp"]
df = pd.read_csv("ml-100k/u.data",sep='\t',names=columns_names)
#print(df.head())
df['user_id'].unique()
df['item_id'].unique()
movie_title=pd.read_csv("ml-100k/u.item",sep="\|",header=None)
movie_title = movie_title[[0,1]]
movie_title.columns=['item_id','title']
#print(movie_title.head())
df = pd.merge(df,movie_title,on="item_id")
#print(df.tail())

df.groupby('title').mean()['rating'].sort_values(ascending=False).head()
df.groupby('title').count()['rating'].sort_values(ascending=False)
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['nums of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
# print(ratings)
ratings.sort_values(by='rating',ascending=False)
plt.figure(figsize=(10,6))
plt.hist(ratings['nums of ratings'],bins=70)
#plt.show()
plt.hist(ratings['rating'],bins=70)
#plt.show()
sns.jointplot(x='rating',y='nums of ratings',data=ratings,alpha=0.5)
movie_matrix = df.pivot_table(index="user_id",columns="title",values="rating")
#print(movie_matrix)
ratings.sort_values('nums of ratings',ascending=False).head()
# starwars_user_rating = movie_matrix['Star Wars (1977)']
# similar_to_sw = movie_matrix.corrwith(starwars_user_rating)
# corr_sw = pd.DataFrame(similar_to_sw,columns=['Correlation'])
# corr_sw.dropna(inplace=True)
# corr_sw.sort_values('Correlation',ascending=False).head(10)
# corr_sw = corr_sw.join(ratings['nums of ratings'])
# print(corr_sw[corr_sw['nums of ratings']>100].sort_values('Correlation',ascending=False))
def predict_movies(movie_name):
    movie_user_rating = movie_matrix[movie_name]
    similar_to_movie = movie_matrix.corrwith(movie_user_rating)
    corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['nums of ratings'])
    predictions = corr_movie[corr_movie['nums of ratings'] > 100].sort_values('Correlation', ascending=False)
    return predictions

predictions = predict_movies("Star Wars (1977)")
print(predictions.head())
