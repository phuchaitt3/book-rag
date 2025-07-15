import streamlit as st
import json
import pandas as pd
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import subprocess
from sentence_transformers import CrossEncoder

# Load the cross-encoder model
@st.cache_resource
def load_cross_encoder():
    try:
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except Exception as e:
        st.error(f"Error loading Cross-Encoder model: {e}")
        return None
    
cross_encoder = load_cross_encoder()

if st.button('Refresh'):
    st.session_state["processed_contexts"] = set()
    st.session_state["chat_history"] = []
    st.rerun()

# Add the utils directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, "utils")
sys.path.append(utils_path)

# --- Global Variables and Model Initialization ---

@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('distiluse-base-multilingual-cased-v1')
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        return None

@st.cache_resource
def load_openai_key():
    try:
        key_path = os.path.join(utils_path, "config.json")
        with open(key_path, "r") as file:
            config = json.load(file)
            return config["openai_api_key"]
    except Exception as e:
        st.error(f"Error loading OpenAI API key: {e}")
        return None

@st.cache_resource
def load_embeddings_and_index():
    try:
        context_embeddings_path = os.path.join(current_dir, 'context_embeddings.npy')
        question_embeddings_path = os.path.join(current_dir, 'question_embeddings.npy')

        context_embeddings_np = np.load(context_embeddings_path)
        question_embeddings_np = np.load(question_embeddings_path)
        embeddings_np = question_embeddings_np
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        return context_embeddings_np, question_embeddings_np, index
    except Exception as e:
        st.error(f"Error loading embeddings or creating FAISS index: {e}")
        return None, None, None

@st.cache_resource
def load_dataset():
    try:
        dataset_path = "./result.xlsx"
        df = pd.read_excel(dataset_path)

        # Check for NaN values in the 'context' column
        nan_contexts = df[df['context'].isna()]
        if not nan_contexts.empty:
            print("Entries with NaN context:")
            print(nan_contexts)

        dataset = df.to_dict(orient="records")
        # st.write(dataset)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []

model = load_model()
openai.api_key = load_openai_key()
context_embeddings, question_embeddings, index = load_embeddings_and_index()
data = load_dataset()

# --- Function Definitions ---

def retrieve_top_k(query, k=5):
    if model is None or index is None or cross_encoder is None:
        st.error("Model, index, or cross-encoder not loaded properly.")
        return []

    # Step 1: Retrieve initial top-k candidates using the bi-encoder
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()
    distances, indices = index.search(np.array([query_embedding_np]), k)

    # Step 2: Prepare pairs for cross-encoder reranking
    candidate_pairs = [(query, st.session_state["dataset"][i]['context']) for i in indices[0]]

    # Step 3: Use cross-encoder to score and rerank the candidates
    scores = cross_encoder.predict(candidate_pairs)
    ranked_indices = np.argsort(scores)[::-1]  # Sort scores in descending order

    # Step 4: Return the reranked results
    results = [st.session_state["dataset"][indices[0][i]] for i in ranked_indices]
    # print(results)
    return results

def generate_answer_chatgpt(conversation_history, model="gpt-4o-mini", max_tokens=500, temperature=0.7):
    if openai.api_key is None:
        st.error("OpenAI API key not loaded properly.")
        return ""

    response = openai.chat.completions.create(
        model=model,
        messages=conversation_history,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature
    )
    answer = response.choices[0].message.content.strip()
    return answer

# --- Initialize Session State ---

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "dataset" not in st.session_state:
    st.session_state["dataset"] = data

if "pdf_book_info" not in st.session_state:
    st.session_state["pdf_book_info"] = {"title": None, "author": None, "year": None}

# --- Sidebar Configuration ---

with st.sidebar:
    st.header("Model Configuration")
    st.session_state["temperature"] = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    st.session_state["max_tokens"] = st.slider("Max tokens", 100, 1000, 500)
    st.session_state["chat_model"] = st.selectbox("Chat Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"], index=0)

# --- Upload PDF File ---

uploaded_pdfs_dir = os.path.join(current_dir, "uploaded_pdfs")
os.makedirs(uploaded_pdfs_dir, exist_ok=True)

def extract_new_data_from_excel(existing_data):
    # Load the updated dataset
    dataset_path = "./result.xlsx"
    df = pd.read_excel(dataset_path)
    updated_data = df.to_dict(orient="records")

    # Determine new entries by comparing with existing data
    new_entries = [entry for entry in updated_data if entry not in existing_data]

    # Extract contexts and questions from new entries
    new_contexts = [entry['context'] for entry in new_entries if 'context' in entry]
    new_questions = [entry['question'] for entry in new_entries if 'question' in entry]

    return new_contexts, new_questions, updated_data

def update_embeddings(new_contexts, new_questions):
    # Load existing embeddings
    context_embeddings_path = os.path.join(current_dir, 'context_embeddings.npy')
    question_embeddings_path = os.path.join(current_dir, 'question_embeddings.npy')

    existing_context_embeddings = np.load(context_embeddings_path)
    existing_question_embeddings = np.load(question_embeddings_path)

    # Generate new embeddings
    new_context_embeddings = model.encode(new_contexts, convert_to_tensor=True).cpu().detach().numpy()
    new_question_embeddings = model.encode(new_questions, convert_to_tensor=True).cpu().detach().numpy()

    # Concatenate new embeddings with existing ones
    updated_context_embeddings = np.concatenate((existing_context_embeddings, new_context_embeddings), axis=0)
    updated_question_embeddings = np.concatenate((existing_question_embeddings, new_question_embeddings), axis=0)

    # Update FAISS index
    index.add(new_question_embeddings)

    # Save updated embeddings
    np.save(context_embeddings_path, updated_context_embeddings)
    np.save(question_embeddings_path, updated_question_embeddings)

    return updated_context_embeddings, updated_question_embeddings

with st.expander("Upload PDF"):
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_file_uploader")

    if pdf_file:
        pdf_path = os.path.join(uploaded_pdfs_dir, pdf_file.name)

        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        st.success(f"Successfully uploaded PDF: {pdf_file.name}")

        if st.button("Process PDF"):
            if pdf_path:
                try:
                    st.write("Running scripts to process PDF and generate Q&A...")
                    subprocess.run(["python", "process_pdf_gg.py", pdf_path], check=True)
                    st.success("Processed PDF with process_pdf_gg.py")

                    subprocess.run(["python", "process_contexts.py"], check=True)
                    st.success("Processed contexts with process_contexts.py")

                    subprocess.run(["python", "gen_QA.py"], check=True)
                    st.success("Generated Q&A with gen_QA.py")

                    # Extract new data and update embeddings
                    new_contexts, new_questions, updated_data = extract_new_data_from_excel(st.session_state["dataset"])
                    context_embeddings, question_embeddings = update_embeddings(new_contexts, new_questions)

                except subprocess.CalledProcessError as e:
                    st.error(f"An error occurred while running the scripts: {e}")
    else:
        st.info("Please upload a PDF file.")

# --- Chatbot Interface ---

st.title("Simple Chatbot About Books")

query = st.text_input("Enter your question:", key="query_input")

if st.button("Submit") and query:
    if len(st.session_state["dataset"]) > 0:
        # Initialize a set to track processed contexts
        if "processed_contexts" not in st.session_state:
            st.session_state["processed_contexts"] = set()

        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = [
                {"role": "system", "content": "You are a helpful assistant that responds in Vietnamese."}
            ]

        conversation_history = st.session_state["conversation_history"]

        # Combine contexts and check for new ones
        combined_context = ""
        metadata_info = ""

        top_k_results = retrieve_top_k(query, k=5)

        for result in top_k_results:
            context = result['context']
            # st.write(st.session_state["processed_contexts"])
            # st.write(context)
            # st.write(context in st.session_state["processed_contexts"])
            if context not in st.session_state["processed_contexts"]:
                # st.write("Run")
                # Add new context to the combined context
                combined_context += context + " "

                # st.write("Result: ", result)
                # st.write("Book title: ", result.get('book_title', 'N/A'))
                # Append metadata for new context
                metadata_info += "\n".join([
                    f"Book Title: {result.get('book_title', 'N/A')}.",
                    f"Authors: {result.get('authors', 'N/A')}.",
                    f"Page Number: {result.get('page_num', 'N/A')}.",
                    f"Publication Time: {result.get('publication_time', 'N/A')}.",
                    f"Heading: {result.get('heading', 'N/A')}."
                ]) + "\n"
                
                # st.write(metadata_info)

                # Mark this context as processed
                st.session_state["processed_contexts"].add(context)

        # st.write("Metadata information:")
        # st.write(metadata_info)

        # Append the question, context, and metadata to the conversation history
        conversation_history.append({
            "role": "user",
            "content": f"Question: {query}\nContext: {combined_context.strip()}\nMetadata: {metadata_info.strip()}\nAnswer: "
        })
        # st.write(f"Conversation history: {conversation_history}")

        answer = generate_answer_chatgpt(
            conversation_history,
            model=st.session_state["chat_model"],
            max_tokens=st.session_state["max_tokens"],
            temperature=st.session_state["temperature"]
        )

        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        # Store the updated conversation history back in session state
        st.session_state["conversation_history"] = conversation_history

        st.session_state["chat_history"].append({"question": query, "answer": answer, "source": "Generated"})
        st.write(f"**Answer:** {answer}")

# --- Display Chat History ---

if st.session_state["chat_history"]:
    st.subheader("Chat history")
    for chat in st.session_state["chat_history"]:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        if "source" in chat:
            st.write(f"**Source:** {chat['source']}")
    
    # Add a button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared.")
        st.rerun()  # Rerun the app to refresh the display