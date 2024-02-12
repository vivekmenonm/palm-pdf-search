import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI

DATA_FOLDER = './data'

def load_pdf_documents(data_folder):
    pdf_files = [fn for fn in os.listdir(data_folder) if fn.endswith('.pdf')]
    loaders = [PyPDFLoader(os.path.join(data_folder, fn)) for fn in pdf_files]
    print(f'{len(loaders)} files loaded')
    return loaders

def combine_documents(loaders):
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents

def embed_texts(documents):
    embeddings = VertexAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    return db

def initialize_retriever(llm, db):
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def retrieve_answer(query):
    start = time.time()

    loaders = load_pdf_documents(DATA_FOLDER)
    documents = combine_documents(loaders)
    db = embed_texts(documents)

    emb_time = time.time()
    print("Embedding took:", emb_time - start)

    db.save_local("engineering_docs")

    llm = VertexAI(
        model_name="text-bison@001",
        project='project-name',
        temperature=0.9,
        top_p=0,
        top_k=1,
        max_output_tokens=256
    )

    qa = initialize_retriever(llm, db)

    result = qa({"query": query})
    filtered_metadata = [doc.metadata for doc in result['source_documents']]
    result_value = result['result']

    end = time.time()
    print("Total time taken:", end-start)
    return result

if __name__ == "__main__":
    main()
