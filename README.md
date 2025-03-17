# Company Knowledge Assistant

## Description

This is a Retrieval-Augmented Generation (RAG) chatbot for synthetic sample data about a fictional insurance tech company called InsureLLM. All company information, products, and metrics are entirely fictional and generated solely for demonstration purposes:. The assistant can answer questions about the company's products, history, and other information stored in a knowledge base.

## Features

- **RAG-based Question Answering**: Uses LangChain and OpenAI's GPT-4o-mini to answer questions based on company knowledge.
- **Vector Database**: Stores and retrieves document chunks using Chroma.
- **Interactive Interface**: User-friendly Gradio interface with suggested questions and company information.
- **Markdown Support**: Renders responses with proper formatting.

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://huggingface.co/spaces/YOUR_USERNAME/insurellm-assistant
   cd insurellm-assistant
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key:**
   - On Hugging Face Spaces: Add your OpenAI API key as a secret named `OPENAI_API_KEY`
   - Locally: Create a `.env` file with `OPENAI_API_KEY=your_api_key_here`

4. **Run the application:**
   ```
   python app.py
   ```

## Using the Assistant

1. Type your question in the text box and press Enter.
2. Use the suggested question buttons for quick access to common queries.
3. The assistant will search the knowledge base and provide relevant answers.

## Customizing the Knowledge Base

To use your own documents:

1. Create a directory structure like:
   ```
   company_knowledgebase/
   ├── category1/
   │   ├── document1.md
   │   └── document2.md
   ├── category2/
   │   └── document3.md
   ```

2. Add your markdown files to the appropriate categories.

3. The system will automatically load, process, and index these documents when started.

## About the Demo Data

The demo includes sample data about a fictional insurance tech company called InsureLLM:

- Founded by Avery Lancaster in 2015
- 200 employees across 12 offices in the US
- 4 main products: Carllm, Homellm, Rellm, and Marketllm
- Over 300 clients worldwide

## Technologies Used

- LangChain for RAG implementation
- OpenAI for embeddings and language model
- Chroma for vector database
- Gradio for the user interface

## License

This project is for demonstration purposes only.