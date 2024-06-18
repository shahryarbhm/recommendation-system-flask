import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def recommend_by_genre(movie_id, movie_data, top_n=5):
    # Get the genres of the reference movie
    reference_movie = movie_data.iloc[movie_id]
    reference_genres = set(reference_movie['genres'].split('|'))
    
    # Calculate the genre similarity for each movie
    genre_similarity_scores = []
    for index, movie in movie_data.iterrows():
        if movie['movieId'] != reference_movie['movieId']:
            movie_genres = set(movie['genres'].split('|'))
            similarity_score = jaccard_similarity(reference_genres, movie_genres)
            genre_similarity_scores.append((movie['movieId'], similarity_score))
    
    # Sort the movies based on genre similarity scores
    genre_similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    return genre_similarity_scores[:top_n]

def recommend_by_tag(movie_id, movie_data, tag_data, top_n=5):
    # Get the tags of the reference movie
    movie_tags_dict = tag_data.groupby('movieId')['tag'].apply(set).to_dict()
    reference_movie_tags = movie_tags_dict.get(movie_id, set())

    # Calculate the tag similarity for each movie
    tag_similarity_scores = []
    for movie_data_id in movie_data['movieId']:
        if movie_id != movie_data_id:
            movie_tags = movie_tags_dict.get(movie_data_id, set())
            similarity_score = jaccard_similarity(reference_movie_tags, movie_tags)
            tag_similarity_scores.append((movie_data_id, similarity_score))

    # Sort the movies based on tag similarity scores
    tag_similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top-N similar movies
    return tag_similarity_scores[:top_n]

def recommend_by_collaborative_filtering(movie_id, rating_data, top_n=5, sample_frac=1.0):
    # Create a user-movie rating matrix
    if sample_frac < 1.0:
        rating_data = rating_data.sample(frac=sample_frac, random_state=1)
    
    movie_ratings = rating_data[rating_data['movieId'] == movie_id]
    
    # Create a sparse user-movie rating matrix
    rating_matrix = pd.pivot_table(movie_ratings, values='rating', index='userId', columns='movieId').fillna(0)
    rating_matrix = csr_matrix(rating_matrix.values)

    
    # Calculate cosine similarity between movies
    similarity_matrix = cosine_similarity(rating_matrix.T)
    
    # Get the index of the reference movie
    try:
        # Get the index of the reference movie in the similarity matrix
        movie_index = movie_ratings['movieId'].unique().tolist().index(movie_id)
    except ValueError:
        print(f"Movie_id not found in rating_data.")
    
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    
    # Sort the movies based on similarity scores (descending order)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top-N similar movies (excluding the reference movie itself)
    top_similar_movies = [movie_ratings['movieId'].unique()[idx] for idx, _ in similarity_scores[1:top_n+1]]
    
    return top_similar_movies

def recommend_by_hybrid(movie_id, movie_data, tag_data, rating_data, genome_scores, genome_tags, top_n=5):
    # Example weights for each recommendation strategy
    weights = {
        'tag': 0.2,
        'genome_tags': 0.2,
        'collaborative_filtering': 0.3,
        'genre': 0.3
    }

    # Assume each recommendation function is modified to return a list of tuples: (movie_id, score)
    content_based_recommendations = [(movie_id, score * weights['collabrative_filtering']) for movie_id, score in recommend_by_collaborative_filtering(movie_id,  rating_data, top_n)]
    genre_recommendations = [(movie_id, score * weights['genre']) for movie_id, score in recommend_by_genre(movie_id, movie_data, top_n)]
    tag_recommendations = [(movie_id, score * weights['tag']) for movie_id, score in recommend_by_tag(movie_id, movie_data, tag_data, top_n)]
    genome_tags_recommendations = [(movie_id, score * weights['genome_tags']) for movie_id, score in recommend_by_genome_tags(movie_id, movie_data, genome_scores, genome_tags, top_n)]

    # Combine all recommendations into a single list
    combined_recommendations = content_based_recommendations +  genome_tags_recommendations + genre_recommendations + tag_recommendations

    # Aggregate scores for the same movie IDs
    aggregated_scores = {}
    for movie_id, score in combined_recommendations:
        if movie_id in aggregated_scores:
            aggregated_scores[movie_id] += score
        else:
            aggregated_scores[movie_id] = score

    # Sort movies based on aggregated scores
    sorted_recommendations = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top-N recommendations
    return sorted_recommendations[:top_n]

def recommend_by_genome_tags(movie_id, movie_data, genome_scores, genome_tags, top_n=5):
    # Get the genome scores for the reference movie
    reference_movie_scores = genome_scores[genome_scores['movieId'] == int(movie_id)]
    
    # Merge the genome scores with the genome tags
    merged_scores = pd.merge(genome_scores, genome_tags, on='tagId')
    
    # Create a movie-tag matrix
    movie_tag_matrix = merged_scores.pivot_table(index='movieId', columns='tag', values='relevance')
    
    # Fill missing values with 0
    movie_tag_matrix.fillna(0, inplace=True)
    
    # Calculate cosine similarity between movies
    similarity_matrix = cosine_similarity(movie_tag_matrix)
    
    # Get the index of the reference movie
    movie_index = movie_tag_matrix.index.get_loc(int(movie_id))
    
    # Get the similarity scores for the reference movie
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    
    # Sort the movies based on similarity scores
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top-N similar movies
    return similarity_scores[1:top_n+1]

# ... (previous code remains the same)
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union