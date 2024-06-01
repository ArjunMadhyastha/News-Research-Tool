# News-Research-Tool
An efficient news information retrieval
![Screenshot 2024-06-01 134212](https://github.com/ArjunMadhyastha/News-Research-Tool/assets/120244775/f6182df5-cd5d-49db-ac17-9a2dd4a3fd32)

# Steps and Features:
-> Upload News Article URLs or text content URLs to fetch the content
-> Process article content LangChain's UnstructuredURL Loader
-> Create embedding vectors using HuggingFace embeddings and store it using 
   FAISS index, a powerful similarity search library, to enable swift and 
   effective retrieval of relevant information
-> Interact with the LLM by inputting queries and receiving answers along with 
   the source links  
# Installation and Running:
-> Install the required dependencies using pip:
* pip install -r requirements.txt
-> Set up your OpenAI API key by creating a .env file in the project root and 
   adding your API
  * HUGGINGFACEHUB_API_TOKEN=your_api_key
-> Run the Streamlit app by executing:
  * streamlit run main.py
