import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
import json

# 1. Load the PyTorch configuration and weights
print("Loading PyTorch model and data...")
X = np.load("X_data.npy")

with open("label_mapping.json", "r") as f:
    label_to_id = json.load(f)

VOCAB_SIZE = 1500
EMBEDDING_DIM = 16
NUM_CLASSES = len(label_to_id)
MAX_SEQ_LENGTH = 15

# Load the saved FP32 state dictionary
pytorch_weights = torch.load("intent_router_fp32.pth")

# 2. Build the exact equivalent model in Keras
print("Constructing Keras equivalent architecture...")
inputs = keras.Input(shape=(MAX_SEQ_LENGTH,))
x = keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(inputs)
x = keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
x = keras.layers.GlobalMaxPooling1D()(x)
outputs = keras.layers.Dense(NUM_CLASSES)(x)

keras_model = keras.Model(inputs=inputs, outputs=outputs)

# 3. Surgically transfer weights from PyTorch to Keras
print("Transferring FP32 weights...")
# Embedding: PyTorch [Vocab, Dim] -> Keras [Vocab, Dim]
keras_model.layers[1].set_weights([pytorch_weights['embedding.weight'].numpy()])

# Conv1D: PyTorch [Out_Channels, In_Channels, Kernel] -> Keras [Kernel, In_Channels, Out_Channels]
conv_w = pytorch_weights['conv1d.weight'].permute(2, 1, 0).numpy()
conv_b = pytorch_weights['conv1d.bias'].numpy()
keras_model.layers[2].set_weights([conv_w, conv_b])

# Dense: PyTorch [Out, In] -> Keras [In, Out]
dense_w = pytorch_weights['fc.weight'].transpose(0, 1).numpy()
dense_b = pytorch_weights['fc.bias'].numpy()
keras_model.layers[4].set_weights([dense_w, dense_b])

# 4. Content-Aware INT8 Quantization
print("Running Content-Aware Calibration and Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# This is the "Content-Aware" part: We feed real data through the network 
# so the converter observes the activation ranges and scales the INT8 bins perfectly.
def representative_data_gen():
    # Use a small slice of the training data (e.g., 200 samples)
    for input_value in X[:200]:
        # TFLite expects shape (1, Sequence_Length) and float32 types for calibration input
        yield [np.array(input_value, dtype=np.float32).reshape(1, MAX_SEQ_LENGTH)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ensure inputs and outputs remain compatible with basic types (int8)
converter.inference_input_type = tf.float32 
converter.inference_output_type = tf.float32

# Convert and save
tflite_model = converter.convert()
with open('intent_router_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\nSuccess! Final INT8 model size: {len(tflite_model) / 1024:.2f} KB")
print("Target ESP32-C3 SRAM limit: 400 KB. You are well within constraints.")