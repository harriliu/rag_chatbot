import os 
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings


# Load environment variables in a file called .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your_api_key')

# select model, use gpt4o-mini for speed and low cost
llm_model="gpt-4o-mini"
db_name ="vector_db"

folders = glob.glob("company_knowledgebase/*")

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

text_loader_kwargs={'autodetect_encoding': True}

# load knowledge database from the folder
documents = []
for folder in folders:
    doc_type =os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents)
print(f"Total number of chunks: {len(chunks)}")
print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")

# call OpenAI embedding
embeddings = OpenAIEmbeddings()

# if there is vector db created, deleted it to prevent duplication
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

#create vector database using Chroma
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# use LangChain to bring the vector db and llm together
# create a chatbot using OpenAI
llm = ChatOpenAI(temperature = 0.7, model_name = llm_model)

# create chat memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# retrieve context information for GPT 4o, using 25 chunk, when search for context
retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

# putting it together: set up the conversation chain with the GPT 4-0 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Add a chatbot UI using Gradio
# wrap the conversation chain into a function for Gradio UI
def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]
    
def process_message(message, history):
    # Add typing indicator
    typing_message = "Assistant is thinking..."
    
    for i in range(0, 3):
        yield typing_message + "." * i
        time.sleep(0.5)
    
    # Actually process the message with your RAG chain
    result = conversation_chain.invoke({"question": message})
    answer = result["answer"]
    
    # Format the answer with markdown support
    formatted_answer = answer
    
    yield formatted_answer

def quick_question(question):
    return question

def create_enhanced_ui():
    with gr.Blocks() as demo:
        # Company header banner
        with gr.Row(elem_classes="company-banner"):
            gr.HTML("<h2>Company Knowledge Assistant</h2>")
        
        with gr.Row():
            # Main column with chat
            with gr.Column(scale=2):
                # Main chat interface
                with gr.Group(elem_classes="chat-window"):
                    chatbot = gr.ChatInterface(
                        process_message,
                        chatbot=gr.Chatbot(
                            height=500,
                            show_copy_button=True,
                            render_markdown=True,
                            elem_id="chatbot"
                        ),
                        title="Ask anything about InsureLLM",
                        description="I can help with questions about our products, company history, or career opportunities.",
                    )
                
                # Quick questions
                with gr.Group():
                    gr.Markdown("### Suggested Questions")
                    with gr.Row():
                        q1 = gr.Button("which contract has the highest revenue?", elem_classes="question-button")
                        q2 = gr.Button("Who founded the company?", elem_classes="question-button")
                        q3 = gr.Button("can you give me a list of engineer in the company?", elem_classes="question-button")
                        q4 = gr.Button("How many clients does InsureLLM have?", elem_classes="question-button")
                    
                    # Connect buttons to chat
                    q1.click(fn=lambda: "which contract has the highest revenue?", outputs=chatbot.textbox)
                    q2.click(fn=lambda: "Who founded the company?", outputs=chatbot.textbox)
                    q3.click(fn=lambda: "can you give me a list of engineer in the company?", outputs=chatbot.textbox)
                    q4.click(fn=lambda: "How many clients does InsureLLM have?", outputs=chatbot.textbox)
    
        
    return demo

# Launch the enhanced UI
enhanced_ui = create_enhanced_ui()
# Launch the app
if __name__ == "__main__":
    enhanced_ui.launch(share=True)