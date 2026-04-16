import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
def creating_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed" , json = {
        "model": "bge-m3",
        "input" : text_list
    })

    # print(r.json())
    embedding = r.json()['embeddings']
    #print(embedding[0:5])
    return embedding

# # for example
# a = creating_embedding("Hy my name is umang")
# print(a)

jsons = os.listdir("newjsons")  # pahela jsons hatu pan after mearg i do chages
# print(jsons)
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f'jsons/{json_file}') as f:
        content = json.load(f)
    
    print(f'Creating embedding for {json_file}')
    embeddings = creating_embedding([c['text'] for c in content['chunks']])   # [] must

    for i,chunk in enumerate(content['chunks']):
        # print(chunk)
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)


# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# print(df)
joblib.dump(df,'embeddings.joblib')
