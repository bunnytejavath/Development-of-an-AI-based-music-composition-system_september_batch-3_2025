from pinecone import Pinecone,ServerlessSpec
from openai import OpenAI
import os

client = OpenAI(api_key = f'{os.getenv('openai')}')
pc = Pinecone(api_key = f'{os.getenv('pinecone')}' )

pc.create_index(
            name = "appcontext",
            dimension = 3072,
            metric = 'cosine',
            spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
)

index = pc.Index("appcontext")

def embed_text(text):
    res = client.embeddings.create(
            input = text,
            model = "text-embedding-3-large"
        )
    return res.data[0].embedding

doc = [
       {'id':'1', 'text':'Python is a programming language.'},
       {'id':'2', 'text':'Pinecone is a vector database for semantic search.'},
       {'id':'3', 'text':'OpenAI creates powerful AI models like GPT.'}
       ]

vectors = [(d['id'],embed_text(d['text']),{'text':d['text']}) for d in doc]
print(vectors)
index.upsert(vectors)

query = "Explain database pinecone"
query_vector = embed_text(query)
search_results = index.query(
    vector = query_vector,
    top_k = 1,
    include_metadata = True
    )
print(search_results)

prompt = f"Answer the question based on context \ncontext:\n{search_results['matches'][0]['metadata']['text']}\nquestion:\n{query}"
print(prompt)
res=client.responses.create(
    model = "gpt-4o-mini",
    input = [{'role':'user','content':prompt}]
    )
print(res.output[0].content[0].text)