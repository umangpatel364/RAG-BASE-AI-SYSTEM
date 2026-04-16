import requests
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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate" , json = {
        "model": "llama3.2",
        "prompt" : prompt,
        "stream" : False
    })
    response = r.json()
    print(response)
    return response
    
df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")
question_embedding = creating_embedding([incoming_query])[0]

# find similarities of question_embedding with other embedding

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# here .vstack use for covert 1d to 2d array and flatten use for write array horizontly

# print(similarities)
top_results= 5
max_indx = similarities.argsort()[::-1][0:top_results]
# it give result asending order so we use -1 so we get desending order 
#0 to top_result means output give top 5 results only

# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[['title','number','text']])

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time to in seconds, end time in seconds, the text at that time:

{new_df[['title','number','start','end','text']].to_json(orient="records")} 
---------------------------------
'{incoming_query}'
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer question related to the course
'''
# orient = records it give list of discnory
with open('promt.txt','w') as f:
    f.write(prompt)

# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'], item['start'], item['end'])


# i comment this two line for thinking...

# response = inference(prompt)['response']
# print(response)

print("Thinking...\n")

response = inference(prompt)['response']

print("\nAnswer:\n")
print(response)

with open("response.txt", "w") as f:
    f.write(response)