import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("techteach")
index.delete(delete_all=True)
print("All vectors deleted from medicalbot")