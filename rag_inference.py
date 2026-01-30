import logging

# Dependencies: pip install langchain-openai langchain-core
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import your custom VectorDatabase
from rag_storage import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_rag_pipeline(question: str):
    """
    Combines VectorDatabase retrieval with LLM generation.
    """
    # 1. Initialize and Load Vector Database
    # This loads the FAISS index we created in the previous step
    db = VectorDatabase()
    db.load()

    # 2. Initialize Local LLM (LM Studio)
    # Ensure you have a chat model loaded in LM Studio (e.g., Gemma, Llama 3, Mistral)
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="local-model", # meta-llama-3.1-8b-instruct, The model name often doesn't matter for local server, but can be specific
        temperature=0.7
    )

    # 3. Retrieve Context
    logger.info(f"Searching for context related to: '{question}'")
    docs = db.search(question, k=3)
    
    if not docs:
        logger.warning("No relevant context found. Falling back to general knowledge.")
        context_text = ""
        template = """You are a helpful assistant. Answer the following question based on your general knowledge.

        Question: {question}
        """
    else:
        # Combine retrieved documents into a single string
        context_text = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Retrieved {len(docs)} chunks of context.")

        # 4. Define the Prompt Template
        template = """You are a helpful assistant. Answer the question based ONLY on the following context:

        {context}

        Question: {question}
        """

    prompt = ChatPromptTemplate.from_template(template)

    # 5. Create and Run the Chain
    # Chain: Prompt -> LLM -> String Output
    chain = prompt | llm | StrOutputParser()
    
    print("\n--- Generating Answer ---")
    response = chain.invoke({
        "context": context_text,
        "question": question
    })
    print(response)

if __name__ == "__main__":
    # Example usage
    run_rag_pipeline("What is FAISS used for?")