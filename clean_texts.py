import os

input_dir = "dataset"
output_dir = "clean_dataset"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    with open(f"{input_dir}/{file}", "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("\n", " ").strip()

    with open(f"{output_dir}/{file}", "w", encoding="utf-8") as f:
        f.write(text)

print("✅ Nettoyage terminé")

