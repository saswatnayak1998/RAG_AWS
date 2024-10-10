import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Check if a query argument is provided
if len(sys.argv) < 2:
    print("[]")  # Return an empty JSON array if no query is provided
    sys.exit(0)

query = sys.argv[1]
print(f"Received query: {query}", file=sys.stderr)  # Log to stderr for Node.js debugging

# Load the FAISS index and metadata
index_file = "aws_docs_faiss.index"
metadata_file = "aws_metadata.json"

# Load the FAISS index
index = faiss.read_index(index_file)

# Load the metadata, which maps index positions to document information
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Check if the metadata length matches the number of vectors in the index
if len(metadata) != index.ntotal:
    print("[]")  # Return an empty array if there's a mismatch
    print("Metadata and FAISS index size mismatch", file=sys.stderr)
    sys.exit(1)

# Load the SentenceTransformer model used for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the query using the same model
query_vector = model.encode([query])[0].astype('float32').reshape(1, -1)

# Perform the similarity search on the index using the query vector
k = 5  # Number of top results to retrieve
distances, indices = index.search(query_vector, k)

# Collect the top 5 results based on metadata (focusing only on the title)
results = []
for i, idx in enumerate(indices[0]):
    if idx < 0 or idx >= len(metadata):
        continue

    # Extract metadata for each result, considering only the title and URL
    entry = metadata[idx]
    result = {
        "title": entry.get("title", "No Title"),
        "url": entry.get("url", "#"),
        "snippet": entry.get("title", "No Title"),  # Using title as the snippet
        "distance": float(distances[0][i]),  # Include distance for debugging or ranking
    }
    results.append(result)

# Output the top 5 results as JSON
print(json.dumps(results))
