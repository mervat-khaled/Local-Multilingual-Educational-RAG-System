import fitz  # this library = pymupdf
from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

import ollama
import os
import time
import json
import requests
from langchain_community.chat_message_histories import ChatMessageHistory


embeddings= OllamaEmbeddings(
    model='granite-embedding:latest'
)


vector_store = None
retriever = None

#First PDF processing function


def process_pdf(file):
    global vector_store, retriever
    if file is None:
        return "No file uploaded."

    try:
        documents = []
        with fitz.open(file.name) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text().strip()  
                
                # Skip completely blank slides to save vector space
                if not text:
                    continue
                
                # Prepend the slide number directly to the text body.
                # This explicitly forces the 1.5B model to see the context source.
                slide_content = f"[Slide {page_num + 1}]\n{text}"
                
                doc = Document(
                    page_content=slide_content,
                    metadata={"source": file.name, "page": page_num + 1}
                )
                documents.append(doc)

        
        chunks = documents 

        vector_store = QdrantVectorStore.from_documents(
            chunks, 
            embeddings,  
            location=":memory:"
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        

        return f"Successfully processed PDF! Loaded {len(chunks)} slides into the vector store. You can now chat about it."
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}"



system_prompt= ''' 

You are a helpful educational assistant. Your task is to answer the user's question using ONLY the retrieved lecture slide excerpts provided below.

Rules:
1. Language Match: If the user asks in Arabic, reply in fluent Arabic. If they ask in English, reply in English.
2. Rely only on the context. If the context does not contain the answer, say exactly: "I cannot find this in the Slides.” or "لم أجد هذا في الملف.” 
3. Do not invent details. Keep answers short, structured, and factual.
4. Maintain an encouraging, clear, and educational tone throughout.
Retrieved Slide Context:
{retrieved_context}

'''

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:1.5b"
file_path = "RAG_chat_detailed_analysis2.json"
memory = ChatMessageHistory()
performance_metrics = []
turn_counter = 1


def ollama_rag_agent(user_input, history=[]):
    global turn_counter, performance_metrics, retriever

    # Retrieve context from vector store if a document has been uploaded
    retrieved_context = ""
    if retriever is not None:
        docs = retriever.invoke(user_input)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])

    # Build messages payload for Ollama
    ollama_messages = [{"role": "system", "content": system_prompt}]

    if retrieved_context:
        ollama_messages.append({
            "role": "system",
            "content": f"--- CONTEXT FROM UPLOADED DOCUMENT ---\n{retrieved_context}\n--------------------------------------"
        })

    # Append conversation history
    for msg in memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        ollama_messages.append({"role": role, "content": msg.content})

    # Append current user prompt
    ollama_messages.append({"role": "user", "content": user_input})
    payload = {
        "model": OLLAMA_MODEL,
        "messages": ollama_messages,
        "options": {
            "num_ctx": 1024,          
        "temperature": 0.2,        
     
            
            
        }
    }

    # Measure response time
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=820) # 12 min safe HTTP timeout
        response.raise_for_status()

        output = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "message" in data and "content" in data["message"]:
                    output += data["message"]["content"]
    except Exception as e:
        output = f"Network Connection Error: {str(e)}. Ensure Ollama container is running at {OLLAMA_URL}."

    end_time = time.time()
    duration_seconds = round(end_time - start_time, 2)

    # Word counts
    user_word_count = len(user_input.split())
    ai_word_count = len(output.strip().split())

    # Save in memory
    memory.add_user_message(user_input)
    memory.add_ai_message(output.strip())

    # Save metrics
    performance_metrics.append({
        "turn_number": turn_counter,
        "question": user_input,
        "answer": output.strip(),
        "metrics": {
            "execution_time_seconds": duration_seconds,
            "user_question_word_count": user_word_count,
            "ai_answer_word_count": ai_word_count,
            "avg_words_per_second": round(ai_word_count / duration_seconds, 2) if duration_seconds > 0 else 0,
            "retrieved_context": retrieved_context
        },
    })
    turn_counter += 1

    # Save JSON after each turn
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(performance_metrics, f, ensure_ascii=False, indent=4)

    return output

if __name__ == "__main__":
    process_pdf()
    ollama_rag_agent()
    
    
      
