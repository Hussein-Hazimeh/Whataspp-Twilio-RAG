from dataclasses import dataclass
from typing import List
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pinecone import Pinecone
from config import (
    OPENAI_API_KEY, 
    PINECONE_API_KEY, 
    PINECONE_ENVIRONMENT, 
    PINECONE_INDEX_NAME
)

# Initialize Pinecone with new client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

@dataclass
class Deps:
    openai: AsyncOpenAI
    pinecone_index: Pinecone.Index

# Initialize the agent
rag_agent = Agent(
    "openai:gpt-4",
    system_prompt=("You are a Helpful Assiatnt Profiocient in Answering concise,factful and to the point asnwers for questions asked based on the Context provided"
                   "You have to Use the `retrievre_tool' to get relevent context and generate response based on the context retrieved"
        #            """You are a grading assistant. Evaluate the response based on:
        # 1. Relevancy to the question
        # 2. Faithfulness to the context
        # 3. Context quality and completeness
        
        # lease grade the following response based on:
        # 1. Relevancy (0-1): How well does it answer the question?
        # 2. Faithfulness (0-1): How well does it stick to the provided context?
        # 3. Context Quality (0-1): How complete and relevant is the provided context?
        
        # Question: {ctx.deps.query}
        # Context: {ctx.deps.context}
        # Response: {ctx.deps.response}
        
        # Also determine if web search is needed to augment the context.
        
        # Provide the grades and explanation in the JSON format with key atrributes 'Relevancy','Faithfulness','Context Quality','Needs Web Search':
        # {"Relevancy": <score>,
        # "Faithfulness": <score>,
        # "Context Quality": <score>,
        # "Needs Web Search": <true/false>,
        # "Explanation": <explanation>,
        # "Answer":<provide response based on the context from the `retrieve' if 'Need Web Search' value is 'false' otherwise Use the `websearch_tool` function to generate the final reaponse}"""
        ),
    deps_type=Deps
)

@rag_agent.tool
async def retrieve(context: RunContext[Deps], query: str) -> str:
    import logging
    # At the top of your rag_agent.py
    LOGGING_ENABLED = False  # Set to True to enable logging
    
    # In your logging setup
    if LOGGING_ENABLED:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)  # Show only error messages and above
    
    logging.info(f'Retrieving context for query: {query}')
    
    try:
        # Generate embedding for the query
        embedding_response = await context.deps.openai.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
        logging.info(f'Generated embedding: {query_embedding}')
        
        # Search Pinecone
        search_results = context.deps.pinecone_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace='The Gourmet Haven'
        )
        logging.info(f'Search results: {search_results}')
        
        # Format results
        contexts = []
        for match in search_results.matches:
            if match.score > 0.7:  # Only include relevant matches
                contexts.append(f"Context (relevance: {match.score:.2f}):\n{match.metadata['text']}")
        
        return "\n\n".join(contexts) if contexts else "No relevant context found."
    except Exception as e:
        logging.error(f'Error during retrieval: {e}')
        return "An error occurred during retrieval."

async def query_rag_agent(question: str) -> str:
    """Query the RAG agent with a question.
    
    Args:
        question: The question to ask
        
    Returns:
        str: The agent's response
    """
    # Initialize dependencies
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    deps = Deps(openai=openai_client, pinecone_index=index)
    
    # Run the agent
    result = await rag_agent.run(
        f"Use the retrieve tool to find relevant information and answer this question: {question}",
        deps=deps
    )
    
    return result.data 

    