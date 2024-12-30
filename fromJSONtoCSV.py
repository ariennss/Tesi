#then replaced with c# code

import json
import csv
import pandas as pd

with open('C:\\dataset tesi\\goodreads_books_romance.json', 'r') as f:
    data = json.load(f)

with open('C:\\dataset tesi\\goodreads_book_authors.json') as authors:
    autori = json.load(authors)
    authors_df = pd.DataFrame(autori)
    dizionario_autori = authors_df.set_index('author_id')['name'].to_dict()

print("data loaded")
# Loop through each book entry (assuming data is a list)
with open('C:\\dataset tesi\\csvfilefrompython.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'title', 'author', 'description', 'image', 'recensioni']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, book in enumerate(data):
        author_id = book['authors'][0]['author_id']
        autore =dizionario_autori.get(author_id, 'Unknown')
        recensioni = int(book['ratings_count'])
        if recensioni > 200:
            writer.writerow({'id': book['book_id'], 'title': book['title'], 'author': autore, 'description': book['description'], 'image': book['image_url'], 'recensioni': recensioni})

