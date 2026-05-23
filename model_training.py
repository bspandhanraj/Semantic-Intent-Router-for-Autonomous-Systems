import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

print("Loading preprocessed data...")
X = np.load("X_data.npy")
y = np.load("y_data.npy")

with open("label_mapping.json", "r") as f:
    label_to_id = json.load(f)

# Reconstruct unique_intents to dynamically set NUM_CLASSES
unique_intents = list(label_to_id.keys())

# --- Hyperparameters ---
VOCAB_SIZE = 1500       
EMBEDDING_DIM = 16      
NUM_CLASSES = len(unique_intents)  
MAX_SEQ_LENGTH = 15     

BATCH_SIZE = 32

# (The rest of your model_training.py code goes here exactly as before)

# 2. Split your data into Train and Validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert arrays to PyTorch Tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Define the Ultra-Lightweight 1D-CNN Model
class EdgeIntentRouter(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_length):
        super(EdgeIntentRouter, self).__init__()
        
        # Embedding Layer: Maps word IDs to small dense vectors
        # Shape: [Batch, Max_Seq_Length] -> [Batch, Max_Seq_Length, Embedding_Dim]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D Convolution: Slides over words to catch local context phrases
        # Using a kernel size of 3 captures 3-word sub-phrases (trigrams)
        # Permute input to [Batch, Embedding_Dim, Max_Seq_Length] for Conv1d
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Global Max Pooling: Extracts the strongest feature across the sequence
        # Reduces shape from [Batch, 16, Max_Seq_Length] to [Batch, 16]
        # This keeps the model input-length agnostic and highly efficient
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dense Classifier Layer
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, Max_Seq_Length]
        x = self.embedding(x)  
        
        # PyTorch Conv1d expects channels first: [Batch, Channels, Length]
        x = x.permute(0, 2, 1) 
        
        x = self.conv1d(x)
        x = self.relu(x)
        
        x = self.global_pool(x).squeeze(-1) # Shape: [Batch, 16]
        
        output = self.fc(x) # Shape: [Batch, Num_Classes]
        return output

# Instantiate the model
model = EdgeIntentRouter(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, MAX_SEQ_LENGTH)
print(model)

# Quick sanity check with a dummy batch
sample_batch = torch.tensor(X[:2], dtype=torch.long)
with torch.no_grad():
    prediction = model(sample_batch)
    print(f"\nSanity check output shape: {prediction.shape} (Expected: [2, {NUM_CLASSES}])")

import torch.optim as optim

# --- Training Setup ---
EPOCHS = 40
LEARNING_RATE = 0.002

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Tracking metrics
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()
        
    train_acc = 100 * correct_train / total_train
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation Phase
    model.eval()
    correct_val = 0
    total_val = 0
    total_val_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()
            
    val_acc = 100 * correct_val / total_val
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save the FP32 weights. This is crucial for the upcoming Quantization step.
torch.save(model.state_dict(), "intent_router_fp32.pth")
print("\nTraining complete! Saved baseline FP32 model to intent_router_fp32.pth")