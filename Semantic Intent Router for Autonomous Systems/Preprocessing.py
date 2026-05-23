import json
from pathlib import Path
import numpy as np
from collections import Counter
import re

dataset_root = Path(r"C:\Users\spand\Downloads\Semantic Intent Router for Autonomous Systems\Dataset")
json_files = list(dataset_root.rglob("*.json"))

texts = []
labels = []

print("Extracting nested data from JSON...")
for file_path in json_files:
    # Added errors='ignore' to bypass corrupted bytes
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
        
        # Drill down into the schema we discovered
        if 'domains' in data:
            for domain in data['domains']:
                if 'intents' in domain:
                    for intent in domain['intents']:
                        # The label is usually the intent's name
                        intent_name = intent.get('name', 'unknown_intent')
                        
                        # The actual sentences could be under various keys depending on the benchmark
                        # We check the most common ones: 'queries', 'utterances', 'texts', 'data'
                        utterances = intent.get('queries') or intent.get('utterances') or intent.get('texts') or intent.get('data', [])
                        
                        for utterance in utterances:
                            # Sometimes utterances are raw strings, sometimes they are objects {"text": "..."}
                            if isinstance(utterance, str):
                                texts.append(utterance)
                                labels.append(intent_name)
                            elif isinstance(utterance, dict) and 'text' in utterance:
                                texts.append(utterance['text'])
                                labels.append(intent_name)

print(f"Extraction complete! Found {len(texts)} command records.")

# Map string labels to integers
unique_intents = list(set(labels))
label_to_id = {label: idx for idx, label in enumerate(unique_intents)}
y = np.array([label_to_id[label] for label in labels])

# --- Tokenizer (ESP32 constraints) ---
VOCAB_SIZE = 1500  
MAX_SEQ_LENGTH = 15 

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

print("Building edge-friendly vocabulary...")
all_words = []
for text in texts:
    all_words.extend(clean_text(text))

word_counts = Counter(all_words)
common_words = word_counts.most_common(VOCAB_SIZE - 2)

word_to_id = {"<PAD>": 0, "<UNK>": 1}
for idx, (word, _) in enumerate(common_words):
    word_to_id[word] = idx + 2

# Save vocab for C++ deployment
with open("edge_vocab.json", "w") as f:
    json.dump(word_to_id, f)
print("Saved edge_vocab.json for microcontroller deployment.")

# Encode text
def encode_text(text, word_to_id, max_length):
    words = clean_text(text)
    sequence = [word_to_id.get(word, 1) for word in words]
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        sequence = sequence + [0] * (max_length - len(sequence))
    return sequence

X = np.array([encode_text(t, word_to_id, MAX_SEQ_LENGTH) for t in texts])

print(f"Feature matrix shape: {X.shape}") 
print(f"Total Unique Intents: {len(unique_intents)}")

# Save the feature matrices as numpy binaries
np.save("X_data.npy", X)
np.save("y_data.npy", y)

# Save the label mapping so we know our classes
with open("label_mapping.json", "w") as f:
    json.dump(label_to_id, f)

print("Saved preprocessed arrays (X_data.npy, y_data.npy, label_mapping.json) to disk.")