from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from recommenders import (
    recommend_by_genre,
    recommend_by_tag,
    recommend_by_collaborative_filtering,
    recommend_by_hybrid,
    recommend_by_genome_tags
)
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# Load movie data from CSV files
movie_data = pd.read_csv('data/movies.csv')
tag_data = pd.read_csv('data/tags.csv')
rating_data = pd.read_csv('data/ratings.csv')
genome_tags = pd.read_csv('data/genome-tags.csv')
genome_scores = pd.read_csv('data/genome-scores.csv')
links = pd.read_csv('data/links.csv')


# Swagger configuration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Movie Recommender API'
    }
)
# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/', methods=['GET'])
def welcome():
    return jsonify('Hello World')
@app.route('/api/recommend/genre', methods=['GET'])
def recommend_genre():
    imdb_id_str = request.args.get('imdb_id')
    imdb_id = int(imdb_id_str.replace('tt','')) 
    movie_id = links[links['imdbId'] == imdb_id]['movieId'].values[0]
    recommendations = recommend_by_genre(movie_id, movie_data)
    recommendations = [movie_id for movie_id, score in recommendations[:]]
    # Convert the movieId in recommendations to imdbId
    imdb_recommendations = []
    for movie in recommendations:
        imdb_id = links[links['movieId'] == movie]['imdbId'].values
        if len(imdb_id) > 0:
            formatted_id = f"tt{int(imdb_id[0]):07d}"
            imdb_recommendations.append(formatted_id)
    
    return jsonify(imdb_recommendations)

@app.route('/api/recommend/tag', methods=['GET'])
def recommend_tag():
    imdb_id_str = request.args.get('imdb_id')
    imdb_id = int(imdb_id_str.replace('tt','')) 
    movie_id = links[links['imdbId'] == imdb_id]['movieId'].values[0]
    recommendations = recommend_by_tag(movie_id, movie_data,tag_data)
    recommendations = [movie_id for movie_id, score in recommendations[:]]
    # Convert the movieId in recommendations to imdbId
    imdb_recommendations = []
    for movie in recommendations:
        imdb_id = links[links['movieId'] == movie]['imdbId'].values
        if len(imdb_id) > 0:
            formatted_id = f"tt{int(imdb_id[0]):07d}"
            imdb_recommendations.append(formatted_id)
    
    return jsonify(imdb_recommendations)

@app.route('/api/recommend/collaborative', methods=['GET'])
def recommend_collaborative():
    imdb_id_str = request.args.get('imdb_id')
    imdb_id = int(imdb_id_str.replace('tt','')) 
    movie_id = links[links['imdbId'] == imdb_id]['movieId'].values[0]
    recommendations = recommend_by_collaborative_filtering(movie_id, rating_data)
    recommendations = [movie_id for movie_id, score in recommendations[:]]
    # Convert the movieId in recommendations to imdbId
    imdb_recommendations = []
    for movie in recommendations:
        imdb_id = links[links['movieId'] == movie]['imdbId'].values
        if len(imdb_id) > 0:
            formatted_id = f"tt{int(imdb_id[0]):07d}"
            imdb_recommendations.append(formatted_id)
    
    return jsonify(imdb_recommendations)

@app.route('/api/recommend/hybrid', methods=['GET'])
def recommend_hybrid():
    imdb_id_str = request.args.get('imdb_id')
    imdb_id = int(imdb_id_str.replace('tt','')) 
    movie_id = links[links['imdbId'] == imdb_id]['movieId'].values[0]
    recommendations = recommend_by_hybrid(movie_id, movie_data=movie_data, tag_data=tag_data, rating_data=rating_data,genome_scores=genome_scores,genome_tags=genome_tags)
    recommendations = [movie_index for movie_index, _ in recommendations[:]]
    # Convert the movieId in recommendations to imdbId
    imdb_recommendations = []
    for movie in recommendations:
        imdb_id = links[links['movieId'] == movie]['imdbId'].values
        if len(imdb_id) > 0:
            formatted_id = f"tt{int(imdb_id[0]):07d}"
            imdb_recommendations.append(formatted_id)
    
    return jsonify(imdb_recommendations)

@app.route('/api/recommend/genome-scores', methods=['GET'])
def recommend_genome_scores():
    imdb_id_str = request.args.get('imdb_id')
    imdb_id = int(imdb_id_str.replace('tt','')) 
    movie_id = links[links['imdbId'] == imdb_id]['movieId'].values[0]
    recommendations = recommend_by_genome_tags(movie_id, movie_data,genome_scores,genome_tags)
    recommendations = [movie_index for movie_index, _ in recommendations[:]]
    # Convert the movieId in recommendations to imdbId
    imdb_recommendations = []
    for movie in recommendations:
        imdb_id = links[links['movieId'] == movie]['imdbId'].values
        if len(imdb_id) > 0:
            formatted_id = f"tt{int(imdb_id[0]):07d}"
            imdb_recommendations.append(formatted_id)
    
    return jsonify(imdb_recommendations)

if __name__ == '__main__':
    app.run()