import os


def load_data(file_path="data/hungarian_english_pairs.txt"):
  """Reads your saved local text file without Hugging Face or pandas."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file not found at: {file_path}")

  pairs = []
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = line.split("<sep>")
      if len(parts) >= 2:
        pairs.append((parts[0].strip(), parts[1].strip()))

  print(
      f"Successfully loaded {len(pairs)} sentence pairs from local disk."
  )
  return pairs