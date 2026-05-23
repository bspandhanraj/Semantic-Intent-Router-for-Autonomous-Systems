import json

with open("edge_vocab.json", "r") as f:
    vocab = json.load(f)

# Sort the vocabulary alphabetically so we can use Binary Search in C++
sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])

with open("vocab.h", "w") as f:
    f.write("#ifndef VOCAB_H\n#define VOCAB_H\n\n")
    f.write("#include <string.h>\n\n")
    f.write("struct VocabEntry {\n    const char* word;\n    int id;\n};\n\n")
    f.write(f"const int VOCAB_DICT_SIZE = {len(sorted_vocab)};\n\n")
    
    # Write the array into Flash memory (PROGMEM)
    f.write("const VocabEntry vocab_table[] = {\n")
    for word, idx in sorted_vocab:
        safe_word = word.replace('"', '\\"')
        f.write(f'    {{"{safe_word}", {idx}}},\n')
    f.write("};\n\n")

    # Add the Binary Search function
    f.write("""
int get_word_id(const char* target_word) {
    int left = 0;
    int right = VOCAB_DICT_SIZE - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int cmp = strcmp(target_word, vocab_table[mid].word);
        if (cmp == 0) return vocab_table[mid].id;
        if (cmp < 0) right = mid - 1;
        else left = mid + 1;
    }
    return 1; // Return 1 for <UNK> (Unknown word) if not found
}
""")
    f.write("\n#endif // VOCAB_H\n")

print("Successfully generated vocab.h!")