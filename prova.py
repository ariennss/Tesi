import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from win32comext.mapi.emsabtags import PR_EMS_AB_INBOUND_SITES_T


def suggestbook():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("model created")
    book_overviews = [
        "dog",
        "cat",
        "parrot",
    ]
    book_embeddings = model.encode(book_overviews)
    print("books encoded")
    user_query = "an animal that flies"
    query_embedding = model.encode([user_query])
    print("input encoded")
    similarity_scores = cosine_similarity(query_embedding, book_embeddings)
    print("calcolata cosine similarity")
    print(similarity_scores)
    ranked_indices = similarity_scores[0].argsort()[::-1]

    # Recommend the most similar overview
    most_similar_overview = book_overviews[ranked_indices[0]]
    print("most similar word: " + most_similar_overview)
    return most_similar_overview

if __name__ == "__main__":

    model = SentenceTransformer('all-mpnet-base-v2')
    print("model created")
    book_overviews = []
    with open('C:\\dataset tesi\\prova.json', 'r') as f:
        data = json.load(f)

    print("data loaded")
    # Loop through each book entry (assuming data is a list)
    for book in data:
        # Check if "description" key exists
        if "description" in book:
            book_overviews.append(book["description"])

    print("fine ciclo")
    #print(book_overviews)
    book_embeddings = model.encode(book_overviews, batch_size=128)
    print("books encoded")
    user_query = "wizard"
    query_embedding = model.encode([user_query])
    print("input encoded")
    similarity_scores = cosine_similarity(query_embedding, book_embeddings)
    print("calcolata cosine similarity")
    print(similarity_scores)
    ranked_indices = similarity_scores[0].argsort()[::-1]

    # Recommend the most similar overview
    most_similar_overview = book_overviews[ranked_indices[0]]
    print("most similar word: " + most_similar_overview)
