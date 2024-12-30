#TODO: rimuovere la punteggiatura

import pandas as pd
import numpy as np
import csv
import gensim.downloader as api
from gensim.models import KeyedVectors

model_path = "C:\\Downloads\\GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("model loaded / created")
df = pd.read_csv("C:\\tesi\\inputforpython.csv")

with open("C:\\dataset tesi\\stopwordsKaggle.txt", 'r') as f:
    stopwords = f.read().splitlines()

with open('C:\\tesi\\embeddedDescriptions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'vector']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    print("nuovo csv aperto")

    for _, row in df.iterrows():
        book_id = row['id']
        description = str(row['description']).lower()

        # Tokenize the description: prendo la description e tiro fuori parola per parola
        words = [word for word in description.split() if word not in stopwords]
        print(words)

        # creo un array di parole vuoto
        words_in_a_description = []

        # check se le parole dentro la description hanno una corrispondenza nel model
        # se esiste, butta nell'array il vettore corrispondente alla parola
        for word in words:
            if word in model.key_to_index:  # Use key_to_index for word existence check
                words_in_a_description.append(model[word])
        # se l'array è ok, ne fa la media per tirarne fuori un embedding unico
        if words_in_a_description:
            embedding = np.mean(words_in_a_description, axis=0)
            print(embedding)
        else:
            # Handle cases where no words are found in the vocabulary
            embedding = np.zeros(model.vector_size)

        writer.writerow({'id': book_id, 'vector': embedding.tolist()})