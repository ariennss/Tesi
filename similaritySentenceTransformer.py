import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import ast

model = SentenceTransformer('all-mpnet-base-v2')

df = pd.read_csv("C:\\dataset tesi\\embeddedromance.csv")

vectors = {}
for _, row in df.iterrows():
    book_id = row['id']
    description_str = row['vector']
    # Remove square brackets and extra whitespace
    description_str = description_str.strip('[]').strip()
    # Split the string by whitespace
    description_list = description_str.split()
    # Convert the list of strings to a NumPy array of floats
    description_array = np.array(description_list, dtype=float)
    vectors[book_id] = description_array


descriptions = [description for book_id, description in vectors.items()]
print("Tipo della lista di descrizioni: ")
print(type(descriptions))
for description in descriptions:
    print(type(description))

user_query = "a truth that might destroy us forever"
query_embedding = model.encode([user_query])
print(query_embedding)

similarity_scores = cosine_similarity(query_embedding, descriptions)
ranked_indices = similarity_scores[0].argsort()[::-1]
most_similar_overview = descriptions[ranked_indices[0]]
print("most similar:")
print(most_similar_overview)

highest_similarity = -1
most_similar_book_id = None

for book_id, description in vectors.items():
    description = description.reshape(1, -1)
    similarity_score = cosine_similarity(query_embedding, description)[0][0]

    if similarity_score > highest_similarity:
        highest_similarity = similarity_score
        most_similar_book_id = book_id

print(f"Most similar book: {most_similar_book_id}")
print(f"Similarity score: {highest_similarity}")