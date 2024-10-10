from flask import Flask, request, jsonify
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Initialize the Flask app
app = Flask(__name__)

# Define file paths for the FAISS index and metadata
INDEX_FILE = "aws_docs_faiss.index"  # FAISS index file
METADATA_FILE = "aws_metadata.json"  # Metadata JSON file

# Step 1: Load the FAISS index and metadata at server startup
print("Loading FAISS index and metadata...")

# Load the FAISS index file into memory
index = faiss.read_index(INDEX_FILE)

# Load the metadata (which contains titles, URLs, and snippets)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

print(f"FAISS index and metadata loaded successfully. Total documents: {len(metadata)}")

# Step 2: Load the SentenceTransformer model used for vectorization
model_name = 'all-MiniLM-L6-v2'
print(f"Loading SentenceTransformer model: {model_name} ...")
model = SentenceTransformer(model_name)
print("SentenceTransformer model loaded.")

# Step 3: Define the search endpoint
@app.route('/api/search', methods=['GET'])
def search():
    # Retrieve the search query from request parameters
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    print(f"Received query: {query}")

    # Step 4: Convert the query into a vector using the same model used for indexing
    query_vector = model.encode([query])[0].astype('float32').reshape(1, -1)

    # Step 5: Perform a similarity search in the FAISS index
    k = 5  # Number of top results to retrieve
    distances, indices = index.search(query_vector, k)

    # Step 6: Collect the top k results along with metadata
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue  # Skip invalid indices

        # Retrieve metadata for each result
        entry = metadata[idx]
        result = {
            "title": entry.get("title", "No Title"),
            "url": entry.get("url", "#"),
            "snippet": entry.get("content", "No content available")[:200] + "..." if entry.get("content") else "No snippet available",
            "distance": float(distances[0][i]),  # Include distance for ranking/debugging
        }
        results.append(result)

    print(f"Returning {len(results)} results for query '{query}'")

    # Step 7: Return the results as a JSON response
    return jsonify({"links": results})


# Start the Flask app if the script is run directly
if __name__ == '__main__':
    # Use 0.0.0.0 to make the app accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)
