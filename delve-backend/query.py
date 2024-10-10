import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print("[]")  
    sys.exit(0)

query = sys.argv[1]
print(f"Received query: {query}", file=sys.stderr)  # Log to stderr for Node.js debugging

# Load the FAISS index and metadata
index_file = "aws_docs_faiss.index"
metadata_file = "aws_metadata.json"

# Load the FAISS index
index = faiss.read_index(index_file)

with open(metadata_file, "r") as f:
    metadata = json.load(f)

if len(metadata) != index.ntotal:
    print("[]")  # Return an empty array if there's a mismatch
    print("Metadata and FAISS index size mismatch", file=sys.stderr)
    sys.exit(1)

model = SentenceTransformer('all-MiniLM-L6-v2')

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
        "snippet": entry.get("title", "No Title"), 
        "distance": float(distances[0][i]),  
    }
    results.append(result)


print(json.dumps(results))
