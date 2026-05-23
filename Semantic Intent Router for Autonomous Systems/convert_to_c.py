import os

model_path = "intent_router_quantized.tflite"

with open(model_path, "rb") as f:
    tflite_model = f.read()

# Convert binary bytes to hex string array
hex_lines = [', '.join([f'0x{b:02x}' for b in tflite_model[i:i+12]]) for i in range(0, len(tflite_model), 12)]

with open("model.h", "w") as f:
    f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
    f.write(f"// Automatically generated C-array from {model_path}\n")
    f.write(f"const unsigned int model_tflite_len = {len(tflite_model)};\n")
    f.write("const unsigned char model_tflite[] __attribute__((aligned(4))) = {\n")
    f.write(",\n".join(hex_lines))
    f.write("\n};\n\n#endif // MODEL_H\n")

print("Successfully generated model.h!")