import pandas as pd
from sentence_transformers import SentenceTransformer
import csv

model = SentenceTransformer('all-mpnet-base-v2')
print("model created")

df = pd.read_csv("C:\\dataset tesi\\romance.csv")
print("CSV loaded")
print(df.dtypes)

book_descriptions = {}
for _, row in df.iterrows():
    book_id = row['id']
    description = str(row['description'])
    book_descriptions[book_id] = description

with open('C:\\dataset tesi\\embeddedromance.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'vector']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for book_id, description in book_descriptions.items():
        print(type(description))
        encoded_description = model.encode(description)
        writer.writerow({'id': book_id, 'vector': encoded_description})
