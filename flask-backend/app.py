from flask import Flask, request, jsonify
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import ollama  

app = Flask(__name__)

INDEX_FILE = "aws_docs_faiss.index" 
METADATA_FILE = "aws_metadata.json" 


print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)
print(f"FAISS index and metadata loaded successfully. Total documents: {len(metadata)}")


model_name = 'all-MiniLM-L6-v2'
print(f"Loading SentenceTransformer model: {model_name} ...")
model = SentenceTransformer(model_name)
print("SentenceTransformer model loaded.")


client = ollama.Client(host="http://localhost:11434")  

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    print(f"Received query: {query}")

    query_vector = model.encode([query])[0].astype('float32').reshape(1, -1)

    k = 5 
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        entry = metadata[idx]
        result = {
            "title": entry.get("title", "No Title"),
            "url": entry.get("url", "#"),
            "snippet": entry.get("content", "No content available")[:200] + "..." if entry.get("content") else "No snippet available",
            "distance": float(distances[0][i]),  
        }
        results.append(result)

    prompt = f"The user has asked the following query: '{query}'.\n\n"
    prompt += "Here are some relevant links and snippets that might be helpful:\n\n"

    for i, link in enumerate(results):
        prompt += f"Link {i + 1}: {link['title']} ({link['url']})\nSnippet: {link['snippet']}\n\n"

    prompt += (
        "Using the query and the information from the above links, provide a concise and informative summary for the user.\n\n"
        "Please make sure to address the original query and include relevant details from the snippets."
    )

    print("###Generating summary using Ollama...")
    try:
        response = client.generate(model="llama2", prompt=prompt)

        summary_content = response["response"]

        summary = summary_content if summary_content else "No summary available."
    except Exception as e:
        print(f"Failed to generate summary with Ollama: {e}")
        summary = "Error generating summary."

    print("Returning results and summary...")

    return jsonify({"links": results, "summary": summary})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
