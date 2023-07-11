from flask import Flask, request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA
from flask_cors import CORS
import os

app = Flask(__name__)

# CORS(app)
embeddings = VertexAIEmbeddings()
# db = FAISS.load_local("faiss_index", embeddings)
db = FAISS.load_local("engineering_docs", embeddings)

def pdf_loader(data_folder):
    pdf_files = [fn for fn in os.listdir(data_folder) if fn.endswith('.pdf')]
    loaders = [PyPDFLoader(os.path.join(data_folder, fn)) for fn in pdf_files]
    print(f'{len(loaders)} files loaded')
    return loaders


@app.route('/embedding-and-query', methods=['POST'])
def embedding_query():
    data_folder = request.form.get('data_folder')
    question = request.form.get('question')

    # Load multiple PDF documents using the pdf_loader function
    loaders = pdf_loader(data_folder)

    # Combine the loaded documents from different loaders
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    embeddings = VertexAIEmbeddings()

    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

    # Split your docs into texts
    texts = text_splitter.split_documents(documents)

    # Embed your texts
    db = FAISS.from_documents(texts, embeddings)

    # Init your retriever. Asking for just 1 document back
    retriever = db.as_retriever()
    llm = VertexAI(
        model_name="text-bison@001",
        project='your-projectid',
        temperature=0.9,
        top_p=0,
        top_k=1,
        max_output_tokens=256
    )

    # We use Vertex PaLM Text API for LLM
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    result = qa({"query": question})
    result_value = result['result']
    return result_value


@app.route('/query', methods=['POST'])
def query():
    # embeddings = VertexAIEmbeddings()
    # db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever()
    llm = VertexAI(
        model_name="text-bison@001",
        project='your-projectid',
        temperature=0.9,
        top_p=0,
        top_k=1,
        max_output_tokens=256
    )

    # We use Vertex PaLM Text API for LLM
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    query = request.form.get('query')
    result = qa({"query": query})
    result_value = result['result']
    filtered_metadata = [doc.metadata for doc in result['source_documents']]
    filtered_data = []

    for doc in result['source_documents']:
        filtered_data.append({
            'metadata': doc.metadata,
            'page_content': doc.page_content
        })
    return {
        'query': query,
        'result': result_value,
        'source_documents':filtered_data
    }

if __name__ == '__main__':
    app.run(debug=True)
