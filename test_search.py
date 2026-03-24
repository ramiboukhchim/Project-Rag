import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1️⃣ Charger le modèle d'embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2️⃣ Charger les chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 3️⃣ Charger l'index FAISS
index = faiss.read_index("tourisme.index")

# 4️⃣ Question de test
question = "Quelle est l'importance historique de Carthage ?"

# 5️⃣ Transformer la question en embedding
question_embedding = model.encode([question]).astype("float32")

# 6️⃣ Recherche des 3 chunks les plus proches
distances, indices = index.search(question_embedding, k=3)

print("\n🔎 Question :", question)
print("\n📄 Passages trouvés :\n")

for i in indices[0]:
    print("-", chunks[i][:200], "...\n")