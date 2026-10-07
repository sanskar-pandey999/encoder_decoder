import pickle

# Load the saved metadata dictionary from your models folder
with open("models/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("--- Training Metadata & Results ---")
for key, value in metadata.items():
    print(f"{key}: {value}")