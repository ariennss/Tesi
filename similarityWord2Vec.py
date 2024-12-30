#TODO: rimuovere la punteggiatura

import pandas as pd
import numpy as np
import csv
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

with open("C:\\dataset tesi\\stopwordsKaggle.txt", 'r') as f:
    stopwords = f.read().splitlines()

model_path = "C:\\Downloads\\GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("model loaded / created")
df = pd.read_csv("C:\\tesi\\embeddedDescriptions.csv")

dictionary_id_vectors = {} #dictionary book_id <--> array di vettori
for index, row in df.iterrows():
    book_id = row['id']
    embedding = np.array(row['vector'].strip('[]').split(','), dtype=float)
    dictionary_id_vectors[book_id] = embedding

print(type(dictionary_id_vectors))

####    tiro fuori l'embedding della frase in input    ##########
input = "Bill Bryson s first travel book, The Lost Continent, was unanimously acclaimed as one of the funniest books in years. In Neither Here nor Therehe brings his unique brand of humour to bear on Europe as he shoulders his backpack, keeps a tight hold on his wallet, and journeys from Hammerfest, the northernmost town on the continent, to Istanbul on the cusp of Asia. Fluent in, oh, at least one language, he retraces his travels as a student twenty years before. Whether braving the homicidal motorist of Paris, being robbed by gypsies in Florence, attempting not to order tripe and eyeballs in a German restaurant, window-shopping in the sex shops of the Reeperbahn or disputing his hotel bill in Copenhagen, Bryson takes in the sights, dissects the culture and illuminates each place and person with his hilariously caustic observations. He even goes to Liechtenstein."
words = [word for word in input.lower().split() if word not in stopwords]
word_vectors = []
for word in words:
    if word in model.key_to_index:  # Use key_to_index for word existence check
                word_vectors.append(model[word])
if word_vectors:
            query_embedding = np.mean(word_vectors, axis=0)
            query_embedding = query_embedding.reshape(1, -1)
            print("tipo e shape della frase in input")
            print(type(query_embedding))
            print(query_embedding.shape)
else:
            # Handle cases where no words are found in the vocabulary
            embedding = np.zeros(model.vector_size)


###### tiro fuori tutti i vettori del corput ##########
descriptions = [description for book_id, description in dictionary_id_vectors.items()]
descriptions = np.array(descriptions)  # Reshape to have each description as a row with 1 column
print("tipo e shape delle descriptions del corpus")
print(type(descriptions))
print(descriptions.shape)

highest_similarity = -1
most_similar_book_id = None

for book_id, description in dictionary_id_vectors.items():
    description = description.reshape(1, -1)
    similarity_score = cosine_similarity(query_embedding, description)[0][0]

    if similarity_score > highest_similarity:
        highest_similarity = similarity_score
        most_similar_book_id = book_id

print(f"Most similar book: {most_similar_book_id}")
print(f"Similarity score: {highest_similarity}")

#### calcolo cosine similarity ####### con formula pre fatta
similarity_scores = cosine_similarity(query_embedding, descriptions)
ranked_indices = similarity_scores[0].argsort()[::-1]
most_similar_overview = descriptions[ranked_indices[0]]
print("most similar:")
print(most_similar_overview)

print("######################################################################################################")
similarities = {}
for book_id, embedding in dictionary_id_vectors.items():
    similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
    similarities[book_id] = similarity

# Find the most similar book
most_similar_book_id = max(similarities, key=similarities.get)
most_similar_book_embedding = dictionary_id_vectors[most_similar_book_id]

print("Most similar book:", most_similar_book_id)
print("Similarity score:", similarities[most_similar_book_id])