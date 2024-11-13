import os
import streamlit as st
import pickle
import time
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import numpy as np
try:
    import faiss
except ImportError as e:
    print("FAISS library is not installed. Please install it using 'pip install faiss-cpu' or 'conda install -c conda-forge faiss-cpu'.")
    raise e
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
# from haystack.components.embedders import HuggingFaceTEIDocumentEmbedder
# from haystack.utils import Secret
from dotenv import load_dotenv
def test():
    load_dotenv()
    urls = []




urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
# Define the file path for the pickle file
file_path = "faiss_store_HuggingFace.pkl"
main_placefolder = st.empty()
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, huggingface_hub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading ... Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Text Splitter ... Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    #
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model = SentenceTransformer(model_name)
    #
    # doc_texts = [doc.page_content for doc in docs]
    embeddings = HuggingFaceEmbeddings()

    # docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    #
    # index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    # Create a FAISS index using the embeddings
    vector_store = FAISS.from_documents(docs, embeddings)

    main_placefolder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)



    # Save the wrapped FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vector_store, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(),return_source_documents=True)
            result = chain({"query": query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["result"])
            print(result.keys())

            sources = result["source_documents"]
            if sources:
                st.subheader("Sources:")
                sources_list = [doc.metadata['source'] for doc in sources]
                # for source in sources_list:
                #     st.write(source)
                st.write(sources_list[0])
                st.write(sources_list[1])
                st.write(sources_list[2])

