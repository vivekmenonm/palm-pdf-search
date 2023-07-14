from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, ElasticVectorSearch
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
import time
start = time.time()

DATA_FOLDER = './test'


def pdf_loader(data_folder=DATA_FOLDER):
    pdf_files = [fn for fn in os.listdir(data_folder) if fn.endswith('.pdf')]
    loaders = [PyPDFLoader(os.path.join(data_folder, fn)) for fn in pdf_files]
    print(f'{len(loaders)} files loaded')
    return loaders

# Load multiple PDF documents using the pdf_loader function
loaders = pdf_loader()

# Combine the loaded documents from different loaders
documents = []
for loader in loaders:
    documents.extend(loader.load())

embeddings = VertexAIEmbeddings()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)
print("embedding....")
emb_time = time.time()
# Embed your texts
db = ElasticVectorSearch.from_documents(texts, embeddings, elasticsearch_url='http://localhost:9200')

print("embedding took:", emb_time - start)

# Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()
llm = VertexAI(
    model_name="text-bison@001",
    project='my-project',
    temperature=0.9,
    top_p=0,
    top_k=1,
    max_output_tokens=256
)

# We use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

query = "Kolkata office email"
result = qa({"query": query})
filtered_metadata = [doc.metadata for doc in result['source_documents']]
result_value = result['result']
print(result_value)
end = time.time()
print("Total time taken:", end-start)