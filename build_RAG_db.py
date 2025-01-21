import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

def build_michelin_database():
    """
    Build and persist the Michelin guide restaurant database.
    This should be run once before starting the application.
    """
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Set up Chroma with persist_directory
    persist_directory = "chroma_db"
    vectorstore = Chroma(
        collection_name="michelin_guide_restaurants",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Load and store data
    michelin_guide_restaurants_path = "dataset/canada_michelin_guide_restaurants_Aug2024.csv"
    loader = CSVLoader(
        file_path=michelin_guide_restaurants_path,
        source_column='Name',
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ['Name', 'Address', 'Location', 'Price', 'Cuisine', 
                          'Longitude', 'Latitude', 'PhoneNumber', 'Url', 
                          'WebsiteUrl', 'Award', 'GreenStar', 
                          'FacilitiesAndServices', 'Description']
        },
    )
    
    print("Loading documents from CSV...")
    documents = loader.load()
    
    print("Adding documents to Chroma...")
    vectorstore.add_documents(documents)
    
    print(f"Database built successfully and stored in {persist_directory}")
    print(f"You can find the SQLite database at {persist_directory}/chroma.sqlite3")