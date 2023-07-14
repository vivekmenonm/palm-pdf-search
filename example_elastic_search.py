from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Connect to Elasticsearch
es = Elasticsearch(hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Create an index for embeddings
index_name = 'embeddings_index'
embedding_mapping = {
    'mappings': {
        'properties': {
            'embedding': {
                'type': 'dense_vector',
                'dims': 768  # Dimensionality of the embeddings
            }
        }
    }
}

# Check if the index exists, and create it if it doesn't
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=embedding_mapping)
    print(f"Created index '{index_name}'")

# Initialize the Sentence Transformer model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Define a list of sentences to be indexed
sentences = [
    'what is the capital of france',
    'Here comes the second sentence.',
    'And this is the third sentence.'
]

# Index the sentence embeddings
for i, sentence in enumerate(sentences):
    embedding = model.encode([sentence])[0].tolist()
    doc = {'embedding': embedding, 'sentence': sentence}  # Include the original sentence in the document
    es.index(index=index_name, id=i, document=doc)
    print(f"Indexed sentence {i+1}")

# Refresh the index to make the changes visible
es.indices.refresh(index=index_name)

# Define a query sentence
query_sentence = 'capital of france'

# Encode the query sentence
query_embedding = model.encode([query_sentence])[0].tolist()

# Perform a dense vector search
query = {
    'query': {
        'script_score': {
            'query': {
                'match_all': {}  # Match all documents
            },
            'script': {
                'source': 'cosineSimilarity(params.queryVector, "embedding") + 1.0',  # Cosine similarity scoring
                'params': {
                    'queryVector': query_embedding
                }
            }
        }
    }
}

# Perform the search
response = es.search(index=index_name, body=query)

# Retrieve the most matching sentence
match = response['hits']['hits'][0]['_source']['sentence']
print(f"Most matching sentence: {match}")