import streamlit as st
import os
import logging
import sys
import requests

# Dependencies: pip install streamlit langchain-openai langchain-community faiss-cpu
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure we can import modules from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_storage import VectorDatabase
except ImportError:
    st.error("Could not import rag_storage. Make sure rag_storage.py is in the same directory.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Chat with LM Studio", page_icon="🤖", layout="wide")

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_db" not in st.session_state:
        # Initialize VectorDatabase
        # We use a persistent folder for the index relative to this script
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
        st.session_state.vector_db = VectorDatabase(index_folder=index_path)
        st.session_state.vector_db.load()
    
    if "base_url" not in st.session_state:
        st.session_state.base_url = "http://localhost:1234/v1"
    
    if "models_map" not in st.session_state:
        st.session_state.models_map = get_available_models(st.session_state.base_url)

def get_available_models(base_url="http://localhost:1234/v1"):
    """Fetch available models from LM Studio and categorize them."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data['data']
            
            categorized = {}
            llms = []
            embeddings = []
            
            for model in models:
                model_id = model['id']
                # LM Studio IDs often look like "publisher/repo/file"
                parts = model_id.split('/')
                name = parts[-1] if len(parts) > 1 else model_id
                
                item = {"id": model_id, "name": name, "details": model}
                
                if "embed" in model_id.lower():
                    embeddings.append(item)
                else:
                    llms.append(item)
            
            if llms:
                categorized["LLM"] = llms
            if embeddings:
                categorized["Embedding Model"] = embeddings
                
            return categorized
    except Exception as e:
        logger.warning(f"Could not fetch models from LM Studio: {e}")
    return {"LLM": [{"id": "local-model", "name": "None"}]}

def get_llm(base_url, model_name, temperature=0.7):
    """Get the LLM instance."""
    return ChatOpenAI(
        base_url=base_url,
        api_key="lm-studio",
        model=model_name,
        temperature=temperature,
        streaming=True
    )

def main():
    initialize_session_state()
    
    st.title("🤖 Local RAG Chat")
    st.markdown("Chat with your documents using **LM Studio** and **LangChain**.")

    # --- Sidebar: Configuration & Ingestion ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Server Selection
        server_type = st.selectbox("Server Type", ["LM Studio", "Ollama", "Custom"], index=0)
        
        if server_type == "LM Studio":
            st.session_state.base_url = "http://localhost:1234/v1"
            new_base_url = st.text_input("Base URL", value=st.session_state.base_url)
        elif server_type == "Ollama":
            st.session_state.base_url = "http://localhost:11434/v1"
            new_base_url = st.text_input("Base URL", value=st.session_state.base_url)
        else:
            new_base_url = st.text_input("Base URL", value=st.session_state.base_url)
            
        if new_base_url != st.session_state.base_url:
            st.session_state.base_url = new_base_url
            st.session_state.models_map = get_available_models(new_base_url)
            st.rerun()
        
        models_map = st.session_state.models_map
        categories = list(models_map.keys())
        
        col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
        with col1:
            if categories:
                selected_category = st.selectbox("Model Category", categories, index=0)
            else:
                selected_category = st.selectbox("Model Category", ["None"], index=0)
        with col2:
            if st.button("🔄", help="Refresh Models"):
                st.session_state.models_map = get_available_models(st.session_state.base_url)
                st.rerun()
        
        # Filter models by category
        if selected_category and selected_category != "None":
            category_models = models_map.get(selected_category, [])
            model_options = {m["name"]: m["id"] for m in category_models}
        else:
            model_options = {}
        
        if model_options:
            selected_model_name = st.selectbox("Select Model", list(model_options.keys()), index=0)
            selected_model = model_options[selected_model_name]
            
            # Display Model Metadata
            selected_item = next((m for m in category_models if m["id"] == selected_model), None)
            if selected_item and "details" in selected_item:
                with st.expander("ℹ️ Model Metadata"):
                    st.caption(f"**Full ID:** {selected_model}")
                    st.json(selected_item["details"])
        else:
            st.selectbox("Select Model", ["None"], index=0)
            selected_model = None
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful assistant.",
            help="Customize the behavior and persona of the AI assistant."
        )
        
        st.divider()
        st.header("📂 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=["txt", "md", "pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Ingest Files", type="primary"):
                progress_bar = st.progress(0, text="Starting ingestion...")
                total_files = len(uploaded_files)
                count = 0
                
                for i, file in enumerate(uploaded_files):
                    try:
                        file_ext = os.path.splitext(file.name)[1].lower()
                        text = ""
                        
                        if file_ext == ".pdf":
                            try:
                                import pypdf
                                pdf_reader = pypdf.PdfReader(file)
                                for page in pdf_reader.pages:
                                    text += page.extract_text() + "\n"
                            except ImportError:
                                st.error("pypdf is required for PDF support. Run: pip install pypdf")
                                continue
                        else:
                            text = file.getvalue().decode("utf-8")
                        
                        st.session_state.vector_db.ingest(
                            text, 
                            metadata={"source": file.name}
                        )
                        count += 1
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / total_files, text=f"Ingesting {file.name} ({i+1}/{total_files})")
                
                progress_bar.empty()
                
                if count > 0:
                    st.session_state.vector_db.save()
                    st.success(f"Successfully ingested {count} files!")
        
        # Display currently ingested files
        current_files = st.session_state.vector_db.get_ingested_files()
        if current_files:
            st.markdown(f"**📚 Knowledge Base ({len(current_files)} files)**")
            with st.expander("View Files"):
                for f in current_files:
                    st.caption(f"• {f}")
        
        st.divider()
        st.subheader("System Status")
        if st.session_state.vector_db.vector_store:
            # Accessing internal index to get count
            try:
                num_docs = st.session_state.vector_db.vector_store.index.ntotal
                st.info(f"Vector Store: Active ({num_docs} chunks)")
            except:
                st.info("Vector Store: Active")
        else:
            st.warning("Vector Store: Empty")

        if st.button("Clear Database", help="Permanently remove all ingested documents"):
            st.session_state.vector_db.clear()
            st.rerun()

        if st.session_state.messages:
            chat_history_text = ""
            for msg in st.session_state.messages:
                chat_history_text += f"{msg['role'].upper()}:\n{msg['content']}\n\n"
            
            st.download_button(
                label="💾 Export Chat History",
                data=chat_history_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # --- Main Chat Interface ---
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Retrieval Step
            with st.status("Thinking...", expanded=False) as status:
                st.write("Searching knowledge base...")
                docs = st.session_state.vector_db.search(prompt, k=3)
                
                if docs:
                    st.write(f"Found {len(docs)} relevant chunks.")
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Show sources
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get("source", "Unknown")
                        st.text(f"Source {i+1} ({source}):\n{doc.page_content[:100]}...")
                    
                    template = system_prompt + """ Answer the question based ONLY on the following context:

                    {context}

                    Question: {question}
                    """
                    status.update(label="Context retrieved!", state="complete")
                else:
                    st.write("No relevant context found.")
                    context_text = ""
                    template = system_prompt + """ Answer the following question based on your general knowledge.

                    Question: {question}
                    """
                    status.update(label="Using general knowledge", state="complete")

            # Generation Step
            if not selected_model:
                st.error("No model selected. Please check configuration.")
                st.stop()

            try:
                llm = get_llm(st.session_state.base_url, selected_model, temperature)
                prompt_template = ChatPromptTemplate.from_template(template)
                chain = prompt_template | llm | StrOutputParser()
                
                # Stream the response
                for chunk in chain.stream({"context": context_text, "question": prompt}):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.info("Ensure LM Studio is running on port 1234.")

if __name__ == "__main__":
    main()
