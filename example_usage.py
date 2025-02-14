from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from openai import AsyncOpenAI

# # Example documents
# texts = [
#     "gormet haven is a lebanese restaurant",
#     "The sky is blue because of Rayleigh scattering of sunlight.",
#     "Water boils at 100 degrees Celsius at sea level.",
#     "The Earth completes one rotation around its axis in approximately 24 hours."
# ]

# metadata = [
#     {"source": "science_book", "topic": "atmosphere"},
#     {"source": "science_book", "topic": "physics"},
#     {"source": "science_book", "topic": "astronomy"}
# ]

# # Index the documents
# index_documents(texts, metadata)

# Query the RAG agent
import asyncio
from rag_agent import query_rag_agent

async def query_pdf_embeddings(query: str, top_k: int = 5):
    """Query existing PDF embeddings from Pinecone database.
    
    Args:
        query: The search query
        top_k: Number of most relevant results to return
    """
    # Initialize OpenAI and Pinecone clients
    openai_client = AsyncOpenAI()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Generate embedding for the query
    embedding_response = await openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding
    
    # Search Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Print results
    print(f"\nTop {top_k} relevant results for query: '{query}'")
    print("-" * 50)
    for match in search_results['matches']:
        print(f"Score: {match.score:.4f}")
        print(f"Metadata: {match.metadata}")
        print("-" * 50)
    
    return search_results

async def main():
    question = "what is Prompt Leaking?"
    answer = await query_rag_agent(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Query your PDF embeddings
    # query = "what is Bob Smith age?"  # Replace with your actual query
    # await query_pdf_embeddings(query)
    
    

if __name__ == "__main__":
    asyncio.run(main())