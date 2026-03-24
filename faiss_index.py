import faiss
import pickle
import numpy as np

# 1️⃣ Charger les embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings).astype("float32")

# 2️⃣ Créer l'index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 3️⃣ Ajouter les embeddings à FAISS
index.add(embeddings)

# 4️⃣ Sauvegarder l'index
faiss.write_index(index, "tourisme.index")

print("✅ Index FAISS créé et sauvegardé")