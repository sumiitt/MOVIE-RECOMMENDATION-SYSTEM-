# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import mysql.connector
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# Load data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

movies = pd.read_csv('Dataset/movies.csv')
ratings = pd.read_csv('Dataset/ratings.csv')

# Merge data
data = pd.merge(ratings, movies, on='movieId')

# Create a pivot table
ratings_matrix = data.pivot_table(index='userId',columns='title', values='rating')

# Define a Flask app
app = Flask(__name__)
@app.route('/contact-us', methods=['POST'])
def contact_us():
    # Get contact form data
    name = request.form['name']
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']

    # Create a connection to the MySQL database
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="@Arjan196",
        database="movierecommend"
    )

    # Create a cursor to execute SQL queries
    cursor = db.cursor()

    # Define the SQL query to insert contact form data
    sql = "INSERT INTO contacts (name, email, subject, message) VALUES (%s, %s, %s, %s)"

    # Execute the query with the data
    values = (name, email, subject, message)
    cursor.execute(sql, values)

    # Commit the changes to the database
    db.commit()

    # Close the database connection and cursor
    cursor.close()
    db.close()

    return """
                <script>alert("Thank you for your message! We will get back to you soon."); 
                window.location.href = '/';</script>
                """
# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/friends')
def home():
    return render_template('home.html')

# Define a route for the recommendation page
@app.route('/alone')
def alone():
    return render_template('alone.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('Contact.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Load the ratings data
    ratings_matrix = pd.read_csv('Dataset/ratings.csv')
    movies = pd.read_csv('Dataset/movies.csv')
    ratings_matrix = ratings_matrix.pivot(index='userId', columns='movieId', values='rating')

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    ratings_matrix = imputer.fit_transform(ratings_matrix)

    # Convert ratings matrix to CSR format
    X = csr_matrix(ratings_matrix)

    # Perform matrix factorization
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_

    # Get the user input
    movie_title = request.form['movie_names']
    movie_index = movies[movies['title'].str.contains(movie_title)].index[0]

    # Get similar movies
    movie_vec = H[:, movie_index].reshape(1, -1)
    sim_scores = cosine_similarity(movie_vec, H.T)
    sim_scores = sim_scores.flatten()
    sim_scores_index = np.argsort(sim_scores)[::-1][1:6]
    recommended_movie_indices = sim_scores_index.tolist()

    # Get recommended movies and their ratings
    recommended_movies = movies[movies.index.isin(recommended_movie_indices)]['title'].tolist()
    recommended_ratings = ratings_matrix[:, recommended_movie_indices].mean(axis=0).tolist()
    recommended_movies_with_ratings = []
    for i in range(len(recommended_movies)):
        recommended_movies_with_ratings.append((recommended_movies[i], f"{recommended_ratings[i]:.1f}"))

    return render_template('Recommendalone.html', movie=movie_title, recommended_movies=recommended_movies_with_ratings)
@app.route('/recommendation', methods=['POST'])
def recommendation():
    # Get genre preferences from the form
    genre1 = request.form['genre1']
    genre2 = request.form['genre2']

    # Get movies that match both genre preferences
    genre1_movies = set(movies[movies['genres'].str.contains(genre1)]['title'])
    genre2_movies = set(movies[movies['genres'].str.contains(genre2)]['title'])
    common_movies = list(genre1_movies.intersection(genre2_movies))

    # Check if common_movies is not empty
    if not common_movies:
        return render_template('no_movies.html')

    # Compute cosine similarity between movies
    common_movies = list(set(common_movies).intersection(
        ratings_matrix.columns))  # keep only movies that are present in the ratings_matrix
    common_ratings_matrix = ratings_matrix[common_movies].fillna(0)
    movie_similarity = cosine_similarity(common_ratings_matrix.T)

    # Get movie recommendations
    recommendations = []
    recommended_movies = set()
    for i, movie in enumerate(common_movies):
        distances = movie_similarity[i]
        similar_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        for j, similar_movie in enumerate(similar_movies):
            movie_idx = similar_movie[0]
            if movies.iloc[movie_idx]['title'] not in recommended_movies:  # check if movie has not been recommended already
                recommendations.append((movie_idx, similar_movie[1], ratings_matrix.loc[:, movie].mean()))  # add movie rating to recommendations
                recommended_movies.add(movies.iloc[movie_idx]['title'])  # add movie to recommended movies

    # Sort recommendations by similarity score
    recommendations = sorted(recommendations, reverse=True, key=lambda x: x[1])

    # Get movie titles and ratings from their indices
    movie_indices = [r[0] for r in recommendations[:5]]
    movie_titles = [movies.iloc[idx]['title'] for idx in movie_indices]
    movie_ratings = [r[2] for r in recommendations[:5]]

    # Render the recommendation page with the recommendations
    return render_template('Recommendation.html', genre1=genre1, genre2=genre2, recommendations=list(zip(movie_titles, movie_ratings)))



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
