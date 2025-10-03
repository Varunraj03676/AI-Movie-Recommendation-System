import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

# Load CSV files
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('rating.csv')

# Merge both files on movieId
data = pd.merge(ratings, movies, on='movieId')
print(data.isnull().sum())

# Create pivot table for collaborative filtering
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# TF-IDF on genres
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

print(get_recommendations('Toy Story (1995)'))

# KNN for user-based recommendations
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.values)
distances, indices = model_knn.kneighbors([user_movie_matrix.iloc[0].values], n_neighbors=4)
print(indices)

# ---------------------- VISUALIZATIONS ----------------------

# 1. Bar Chart - Top 10 most rated movies
top_movies = data['title'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_movies.values, y=top_movies.index, palette='viridis')
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Ratings Count")
plt.ylabel("Movie Title")
plt.show()

# 2. Histogram - Distribution of ratings
plt.figure(figsize=(8,5))
plt.hist(data['rating'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 3. Pie Chart - Proportion of rating values
rating_counts = data['rating'].value_counts().sort_index()
plt.figure(figsize=(6,6))
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Rating Value Distribution')
plt.axis('equal')
plt.show()

# 4. Line Graph - Average rating per year (optional, if timestamp available)
if 'timestamp' in data.columns:
    data['year'] = pd.to_datetime(data['timestamp'], unit='s').dt.year
    yearly_avg = data.groupby('year')['rating'].mean()
    plt.figure(figsize=(10,5))
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o', color='coral')
    plt.title('Average Movie Rating by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.grid(True)
    plt.show()
