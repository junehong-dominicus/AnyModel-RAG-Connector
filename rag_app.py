import streamlit as st
import os
import logging
import sys
import requests
import tiktoken
import json

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

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    return {}

def save_config():
    config = {
        "base_url": st.session_state.get("base_url"),
        "embedding_model": st.session_state.get("embedding_model"),
        "llm_model": st.session_state.get("llm_model"),
        "temperature": st.session_state.get("temperature"),
        "system_prompt": st.session_state.get("system_prompt"),
        "system_prompt_presets": st.session_state.get("system_prompt_presets"),
        "current_preset_name": st.session_state.get("current_preset_name")
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

def initialize_session_state():
    """Initialize session state variables."""
    config = load_config()

    if "base_url" not in st.session_state:
        st.session_state.base_url = config.get("base_url", "http://localhost:1234/v1")
    
    if "models_map" not in st.session_state:
        st.session_state.models_map = get_available_models(st.session_state.base_url)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = config.get("embedding_model")
        if not st.session_state.embedding_model:
            embed_models = st.session_state.models_map.get("Embedding Model", [])
            if embed_models:
                st.session_state.embedding_model = embed_models[0]["id"]
            else:
                st.session_state.embedding_model = "text-embedding-embeddinggemma-300m-qat"

    if "vector_db" not in st.session_state:
        # Initialize VectorDatabase
        # We use a persistent folder for the index relative to this script
        safe_name = st.session_state.embedding_model.replace("/", "_").replace("\\", "_")
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"faiss_index_{safe_name}")
        st.session_state.vector_db = VectorDatabase(index_folder=index_path, embedding_model=st.session_state.embedding_model)
        st.session_state.vector_db.load()

    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"input": 0, "output": 0}

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = config.get("llm_model")
        
    if "temperature" not in st.session_state:
        st.session_state.temperature = config.get("temperature", 0.7)
        
    if "system_prompt_presets" not in st.session_state:
        st.session_state.system_prompt_presets = config.get("system_prompt_presets", {"Default": "You are a helpful assistant."})

    if "current_preset_name" not in st.session_state:
        st.session_state.current_preset_name = config.get("current_preset_name", "Default")

    if "system_prompt" not in st.session_state:
        # Sync with current preset
        st.session_state.system_prompt = st.session_state.system_prompt_presets.get(st.session_state.current_preset_name, "You are a helpful assistant.")

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
        
        col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
        with col_url:
            if server_type == "LM Studio":
                st.session_state.base_url = "http://localhost:1234/v1"
                new_base_url = st.text_input("Base URL", value=st.session_state.base_url)
            elif server_type == "Ollama":
                st.session_state.base_url = "http://localhost:11434/v1"
                new_base_url = st.text_input("Base URL", value=st.session_state.base_url)
            else:
                new_base_url = st.text_input("Base URL", value=st.session_state.base_url)

        with col_btn:
            if st.button("🔄", help="Refresh Models"):
                st.session_state.models_map = get_available_models(st.session_state.base_url)
                st.rerun()
            
        if new_base_url != st.session_state.base_url:
            st.session_state.base_url = new_base_url
            st.session_state.models_map = get_available_models(new_base_url)
            save_config()
            st.rerun()
        
        models_map = st.session_state.models_map
        
        # Model Selection
        st.subheader("Embedding")
        embed_models = models_map.get("Embedding Model", [])
        if embed_models:
            embed_options = {m["name"]: m["id"] for m in embed_models}
            
            # Find current selection index
            current_id = st.session_state.embedding_model
            current_name = next((k for k, v in embed_options.items() if v == current_id), list(embed_options.keys())[0])
            
            selected_embed_name = st.selectbox("Select Embedding Model", list(embed_options.keys()), index=list(embed_options.keys()).index(current_name), key="embed_model_select")
            selected_embed_id = embed_options[selected_embed_name]
            
            if selected_embed_id != st.session_state.embedding_model:
                st.session_state.embedding_model = selected_embed_id
                # Re-initialize DB with new model path
                st.session_state.pop("vector_db")
                save_config()
                st.rerun()
            
            # Embedding Metadata
            selected_embed_item = next((m for m in embed_models if m["id"] == selected_embed_id), None)
            if selected_embed_item and "details" in selected_embed_item:
                with st.expander("ℹ️ Embedding Metadata"):
                    st.caption(f"**Full ID:** {selected_embed_id}")
                    st.json(selected_embed_item["details"])

        st.subheader("LLM")
        llm_models = []
        for category, models in models_map.items():
            if category != "Embedding Model":
                llm_models.extend(models)
        
        model_options = {m["name"]: m["id"] for m in llm_models}
        
        if model_options:
            # Determine index based on saved config
            current_llm_id = st.session_state.get("llm_model")
            current_index = 0
            if current_llm_id:
                current_name = next((k for k, v in model_options.items() if v == current_llm_id), None)
                if current_name:
                    current_index = list(model_options.keys()).index(current_name)

            selected_model_name = st.selectbox("Select LLM Model", list(model_options.keys()), index=current_index)
            selected_model = model_options[selected_model_name]
            
            if selected_model != st.session_state.get("llm_model"):
                st.session_state.llm_model = selected_model
                save_config()
            
            # Display Model Metadata
            selected_item = next((m for m in llm_models if m["id"] == selected_model), None)
            if selected_item and "details" in selected_item:
                with st.expander("ℹ️ LLM Metadata"):
                    st.caption(f"**Full ID:** {selected_model}")
                    st.json(selected_item["details"])
        else:
            st.selectbox("Select Model", ["None"], index=0)
            selected_model = None
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
            save_config()
        
        st.subheader("System Prompt")
        
        # Preset Selection
        presets = st.session_state.system_prompt_presets
        current_preset = st.session_state.current_preset_name
        
        # Fallback if current preset is missing
        if current_preset not in presets:
            current_preset = "Default"
            if "Default" not in presets:
                presets["Default"] = "You are a helpful assistant."
            st.session_state.current_preset_name = current_preset
            
        selected_preset = st.selectbox("Preset", list(presets.keys()), index=list(presets.keys()).index(current_preset))
        
        if selected_preset != st.session_state.current_preset_name:
            st.session_state.current_preset_name = selected_preset
            st.session_state.system_prompt = presets[selected_preset]
            save_config()
            st.rerun()

        system_prompt = st.text_area(
            "Prompt Content",
            value=st.session_state.system_prompt,
            help="Customize the behavior and persona of the AI assistant."
        )
        if system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = system_prompt
            st.session_state.system_prompt_presets[selected_preset] = system_prompt
            save_config()
            
        with st.expander("Manage Presets"):
            new_preset_name = st.text_input("New Preset Name")
            col_add, col_del = st.columns(2)
            if col_add.button("Save New"):
                if new_preset_name and new_preset_name not in presets:
                    st.session_state.system_prompt_presets[new_preset_name] = system_prompt
                    st.session_state.current_preset_name = new_preset_name
                    save_config()
                    st.rerun()
            if col_del.button("Delete Current"):
                if selected_preset != "Default":
                    del st.session_state.system_prompt_presets[selected_preset]
                    st.session_state.current_preset_name = "Default"
                    st.session_state.system_prompt = st.session_state.system_prompt_presets["Default"]
                    save_config()
                    st.rerun()
        
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
                
                existing_files = st.session_state.vector_db.get_ingested_files()
                
                for i, file in enumerate(uploaded_files):
                    if file.name in existing_files:
                        st.warning(f"Skipping {file.name} - already exists.")
                        progress_bar.progress((i + 1) / total_files, text=f"Skipped {file.name} ({i+1}/{total_files})")
                        continue

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
            with st.expander("Manage Files"):
                for f in current_files:
                    col1, col2 = st.columns([4, 1])
                    col1.caption(f"📄 {f}")
                    if col2.button("🗑️", key=f"del_{f}", help=f"Delete {f}"):
                        st.session_state.vector_db.delete_by_metadata(source=f)
                        st.session_state.vector_db.save()
                        st.rerun()
        
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
            
        # Token Usage Display
        total_tokens = st.session_state.token_usage["input"] + st.session_state.token_usage["output"]
        st.info(f"Token Usage: {total_tokens} (In: {st.session_state.token_usage['input']}, Out: {st.session_state.token_usage['output']})")

        if st.button("Reset Token Usage", help="Reset the token counter to zero"):
            st.session_state.token_usage = {"input": 0, "output": 0}
            st.rerun()

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
                
                # Update Token Usage
                try:
                    encoder = tiktoken.get_encoding("cl100k_base")
                    # Approximate input tokens (System + Context + Question)
                    input_text = f"{system_prompt} {context_text} {prompt}"
                    input_tokens = len(encoder.encode(input_text))
                    output_tokens = len(encoder.encode(full_response))
                    
                    st.session_state.token_usage["input"] += input_tokens
                    st.session_state.token_usage["output"] += output_tokens
                except Exception as e:
                    logger.warning(f"Failed to count tokens: {e}")
                
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.info("Ensure LM Studio is running on port 1234.")

if __name__ == "__main__":
    main()
