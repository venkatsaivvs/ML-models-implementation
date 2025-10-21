#lets create a dataframe with an id videoid, text description, from cricket domain.
import pandas as pd
import numpy as np
video_names = [
    "Virat Kohli's 100th century",
    "MS Dhoni's 100th century",
    "Rohit Sharma's 100th century",
    "Sachin Tendulkar's 100th century",
    "Yuvraj Singh's 100th century",
    "Suresh Raina's 100th century",
    "Ravindra Jadeja's 100th century",
    "Harbhajan Singh's 100th century",
    "Zaheer Khan's 100th century",
    "Ashish Nehra's 100th century",
]   

df = pd.DataFrame({
    'id': np.arange(1, 11),
    'text_description': video_names,
})

print(df)

#lets create embeddings for these from Bert model
from torch import cosine_similarity
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

df['bert_embedding'] = df['text_description'].apply(get_bert_embedding)
print(df)

#save df to csv
df.to_csv('cricket_videos.csv', index=False)

#lets create a function to search for a query in the dataframe
def search_df(query):
    search_embedding = get_bert_embedding(query)
    df['similarity'] = df['bert_embedding'].apply(lambda x: cosine_similarity(x, search_embedding))
    return df.sort_values(by='similarity', ascending=False)

print(search_df("Virat Kohli's 100th century"))

#nearest neighbour search
from sklearn.neighbors import NearestNeighbors

def nearest_neighbour_search(query):
    search_embedding = get_bert_embedding(query)
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(df['bert_embedding'])
    return nn.kneighbors([search_embedding])

print(nearest_neighbour_search("Virat Kohli's 100th century"))