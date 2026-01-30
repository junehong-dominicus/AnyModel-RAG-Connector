# AnyModel-RAG-Connector
AnyModel‑RAG‑Connector is a flexible, multi‑backend Retrieval‑Augmented Generation framework designed to work seamlessly with any LLM provider. Whether you're running local models through LM Studio or Ollama, or cloud‑based models from OpenAI, Gemini, or others, this connector unifies them under a single, consistent RAG workflow.

It provides a clean abstraction layer for embedding, retrieval, and chat orchestration, allowing developers to switch or mix model providers without rewriting their application logic. The result is a portable, modular RAG stack that adapts to your environment—local, cloud, or hybrid.

Key capabilities include:

- Unified interface for multiple LLM backends
- Pluggable embedding and vector store components
- Consistent RAG pipeline across providers
- Easy switching between local and cloud models
- Developer‑friendly architecture for experimentation and production use

AnyModel‑RAG‑Connector is built for developers who want freedom: freedom to choose models, swap providers, and evolve their RAG stack without vendor lock‑in.

## How to Run

### Prerequisites

1.  **Python 3.8+**
2.  **LLM Backend**:
    *   **LM Studio**: Start the local server on port `1234`.
    *   **Ollama**: Ensure the service is running (default port `11434`).

### Installation

1.  Clone the repository and navigate to the project folder.

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run rag_app.py
    ```

2.  Open your browser at `http://localhost:8501`. Configure your server settings in the sidebar and start chatting with your documents.
