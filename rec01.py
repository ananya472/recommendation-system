import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import faiss  # Import Faiss for ANN
from sklearn.metrics import precision_score, recall_score, f1_score

# Load CSV data
content_df = pd.read_csv(r'C:\Users\anany\Desktop\content rec\Content.csv')
view_history_df = pd.read_csv(r'C:\Users\anany\Desktop\content rec\user_history.csv')
user_df = pd.read_csv(r'C:\Users\anany\Desktop\content rec\Users.csv')
interest_df = pd.read_csv(r'C:\Users\anany\Desktop\content rec\Interests.csv')

# Load pre-trained Word2Vec embeddings
word2vec_model = KeyedVectors.load_word2vec_format(r'C:\Users\anany\Desktop\content rec\GoogleNews-vectors-negative300.bin', binary=True)

# Function to get sentence embeddings (averaging word embeddings)
def get_sentence_embedding(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  # Handle out-of-vocabulary words
    return np.mean(word_vectors, axis=0)

# Function to recommend content based on user interests
def recommend_content(user_id):
    # Get user interests
    user_interests = interest_df[interest_df['User_Id'] == user_id]['Interest'].tolist()

    # Get viewed content IDs for the user
    viewed_content_ids = set(view_history_df[view_history_df['user_id'] == user_id]['content_id'].tolist())

    # Initialize an empty DataFrame for recommendations
    all_recommendations = pd.DataFrame()

    # Iterate over all user interests
    for interest in user_interests:
        # Filter content based on the current interest
        filtered_content = content_df[content_df['Category'] == interest]
        
        # Remove already viewed content
        filtered_content = filtered_content[~filtered_content['Video Id'].isin(viewed_content_ids)]
        
        # Use 'Description' or fallback to 'Stream Name'
        if 'Description' not in filtered_content.columns:
            filtered_content['Description'] = filtered_content['Stream Name']
        
        # Get sentence embeddings for all filtered content
        filtered_content['embedding'] = filtered_content['Description'].apply(lambda x: get_sentence_embedding(x, word2vec_model))

        # Prepare Faiss index
        embeddings = np.vstack(filtered_content['embedding'].to_numpy())
        dim = embeddings.shape[1]  # Dimensionality of the embeddings
        index = faiss.IndexFlatL2(dim)  # Using L2 distance for similarity
        index.add(embeddings.astype(np.float32))  # Add embeddings to the index
        
        # Get user embedding
        user_embedding = np.mean(embeddings, axis=0, keepdims=True).astype(np.float32)

        # Search for the nearest neighbors using Faiss
        k = 5  # Number of recommendations to retrieve
        distances, indices = index.search(user_embedding, k)

        # Get top recommendations based on indices
        top_recommendations = filtered_content.iloc[indices[0]].copy()  # Retrieve the corresponding rows
        top_recommendations['similarity'] = 1 - (distances[0] / np.max(distances[0]))  # Normalize distances to similarity scores
        
        # Append to all recommendations
        all_recommendations = pd.concat([all_recommendations, top_recommendations], ignore_index=True)

    # Remove duplicates and sort by similarity
    all_recommendations = all_recommendations.drop_duplicates(subset='Video Id').sort_values(by='similarity', ascending=False).head(10)

    return user_interests, all_recommendations[['Video Id', 'Stream Name', 'Category', 'Likes', 'Comments', 'Views']], all_recommendations['Video Id'].tolist()

# Function to evaluate recommendations
def evaluate_recommendations(user_id, recommended_items):
    user_interests = interest_df[interest_df['User_Id'] == user_id]['Interest'].tolist()
    ground_truth_items = view_history_df[view_history_df['user_id'] == user_id]['content_id'].tolist()
    ground_truth_items += content_df[content_df['Category'].isin(user_interests)]['Video Id'].tolist()
    y_true = [1 if item in ground_truth_items else 0 for item in recommended_items]
    y_pred = [1] * len(recommended_items)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

# Example usage for user with ID 16
user_id_to_recommend = 20
user_interests, recommended_content, recommended_items = recommend_content(user_id_to_recommend)

print(f"User Interests for User ID {user_id_to_recommend}: {user_interests}")
print(f"Recommendations for User ID {user_id_to_recommend}:")
print(recommended_content)

precision, recall, f1 = evaluate_recommendations(user_id_to_recommend, recommended_items)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

