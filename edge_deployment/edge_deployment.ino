#include "model.h"
#include "vocab.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TFLite globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Allocate memory for the neural network operations (32 KB is plenty for our 1D-CNN)
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Our fixed sequence length from Python preprocessing
const int MAX_SEQ_LENGTH = 15;

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for serial monitor to open
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  Serial.println("Loading Edge Intent Router...");

  // Load the C-array model
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model schema mismatch!");
    return;
  }

  // Pull in all operation sets (Conv1D, Dense, MaxPool, etc.)
  static tflite::AllOpsResolver resolver;

  // Build the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Assign pointers to input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Model loaded and memory allocated successfully.");
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming text command from the Serial Monitor
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toLowerCase(); // Ensure lowercase to match our python preprocessing

    if (command.length() == 0) return;
    
    Serial.print("\n[Received]: ");
    Serial.println(command);

    // Array to hold our tokenized integers, initialized to 0 (<PAD>)
    float sequence[MAX_SEQ_LENGTH] = {0}; 
    int word_count = 0;
    
    // Simple String Splitting (Tokenizer)
    int start_idx = 0;
    int space_idx = command.indexOf(' ');
    
    while (space_idx != -1 && word_count < MAX_SEQ_LENGTH) {
      String word = command.substring(start_idx, space_idx);
      sequence[word_count] = get_word_id(word.c_str());
      word_count++;
      
      start_idx = space_idx + 1;
      space_idx = command.indexOf(' ', start_idx);
    }
    
    // Grab the last word
    if (start_idx < command.length() && word_count < MAX_SEQ_LENGTH) {
      String word = command.substring(start_idx);
      sequence[word_count] = get_word_id(word.c_str());
    }

    // Print the tokenized sequence
    Serial.print("[Tokenized]: [ ");
    for(int i=0; i < MAX_SEQ_LENGTH; i++) {
      Serial.print((int)sequence[i]); Serial.print(" ");
      input->data.f[i] = sequence[i]; // Load into model
    }
    Serial.println("]");

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Inference failed!");
      return;
    }

    // Extract the predicted class
    int num_classes = output->dims->data[1];
    float max_score = -100.0;
    int predicted_class = 0;

    for (int i = 0; i < num_classes; i++) {
      if (output->data.f[i] > max_score) {
        max_score = output->data.f[i];
        predicted_class = i;
      }
    }

    Serial.print("[Prediction]: Intent ID ");
    Serial.print(predicted_class);
    Serial.print(" (Confidence: ");
    Serial.print(max_score);
    Serial.println(")");
  }
}
