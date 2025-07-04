# --- 1. Import Necessary Libraries ---
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

# --- 2. Load Environment Variables ---
# This line loads the OPENAI_API_KEY from your .env file
load_dotenv()

# We need to tell LangChain where to find our vector database
# We'll create it in a folder named 'db'
PERSIST_DIRECTORY = 'db'

# --- 3. Load and Process the Document ---
# This function loads the PDF, splits it into chunks, creates embeddings,
# and saves them to the vector database.
def setup_vector_database():
    print("Setting up the vector database...")
    
    # Path to your PDF document
    # Make sure 'sample_document.pdf' is inside a 'docs' folder
    file_path = os.path.join('docs', 'sample_document.pdf')
    
    # Load the document using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the document.")

    # Split the document into smaller chunks for processing
    # This is important for the model's context limit and for finding relevant info
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split the document into {len(texts)} text chunks.")

    # Create embeddings for the text chunks using OpenAI
    # This turns your text into numerical vectors
    embeddings = OpenAIEmbeddings()

    # Create a Chroma vector store and save the embeddings
    # This is the "database" that stores your document's knowledge
    # The 'persist_directory' tells Chroma where to save the files on your computer
    print("Creating and persisting the vector store...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    print("Setup complete. The vector database is ready.")
    return db

# --- 4. Query the Database ---
def ask_question(question):
    print(f"\nAsking the question: {question}")
    
    # Initialize the OpenAI embeddings and the Chroma database from the saved files
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    # Create a "retriever"
    # This object knows how to fetch relevant documents from the database based on a query
    retriever = db.as_retriever(search_kwargs={"k": 3}) # "k: 3" means it will retrieve the top 3 most relevant chunks

    # Create the Q&A chain
    # This chain combines the language model (OpenAI) with the retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7), # Using OpenAI. 'temperature' controls creativity.
        chain_type="stuff", # "stuff" is the simplest method: it just "stuffs" the retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True # This lets us see which chunks were used for the answer
    )

    # Execute the query and get the result
    result = qa_chain({"query": question})
    
    print("\n--- Answer ---")
    print(result["result"])
    print("\n--- Sources ---")
    for doc in result["source_documents"]:
        print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...") # Print the page number and a snippet of the source

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # This checks if the database is already built. If not, it builds it.
    if not os.path.exists(PERSIST_DIRECTORY):
        setup_vector_database()
    
    # Now, you can ask questions.
    my_question = "What is the main topic of this document?" # <--- CHANGE THIS TO YOUR QUESTION
    ask_question(my_question)

    another_question = "Can you summarize the introduction?" # <--- ADD ANOTHER QUESTION
    ask_question(another_question)