# Turkish Morphological Tokenizer

A high-performance morphological tokenizer for Turkish text, available in both Python and Rust implementations. The tokenizer handles root words, suffixes, and special cases like uppercase letters and punctuation.

## Features

- Morphological tokenization of Turkish text
- Root word and suffix detection
- Special handling of uppercase letters with `<UPCL>` tokens
- Byte-pair encoding (BPE) fallback for unknown tokens
- Support for Turkish characters
- JSON output format
- Parallel processing support (Rust implementation)
- Identical output format in both implementations

## Prerequisites

### For Python Implementation
- Python 3.6+
- Required dictionary files:
  - `kokler_v04.json`: Root words dictionary
  - `ekler_v04.json`: Suffixes dictionary
  - `bpe_v02.json`: BPE tokens dictionary

### For Rust Implementation
- Rust (latest stable version)
- Required dependencies (specified in `Cargo.toml`):
  - `serde` and `serde_json` for JSON handling
  - `regex` for text processing
  - `rayon` for parallel processing
- Same dictionary files as Python implementation

## Installation

### Python Implementation
No installation needed. Just ensure you have the required JSON files in the same directory as the script.

### Rust Implementation
1. Clone the repository:
```bash
git clone <repository-url>
cd turkish_tokenizer
```

2. Build the release version:
```bash
cargo build --release
```

The compiled binary will be available at `target/release/turkish_tokenizer`.

## Usage

### Python Implementation

```python
from turkish_tokenizer import tokenize

# Example usage
text = "Kitabı ve defterleri getirn, YouTube"
result = tokenize(text)
print(result)
```

### Rust Implementation (Command Line)

```bash
./target/release/turkish_tokenizer "Kitabı ve defterleri getirn, YouTube"
```

For text with multiple words, you can either:
1. Use quotes around the entire text:
```bash
./target/release/turkish_tokenizer "Bu bir örnek cümledir"
```

2. Or provide the words as separate arguments:
```bash
./target/release/turkish_tokenizer Bu bir örnek cümledir
```

### Output Format

Both implementations produce identical JSON output with two arrays:
1. `tokens`: The tokenized strings
2. `ids`: The corresponding token IDs

Example output:
```json
{
  "tokens": [
    "<UPCL>",
    "kitab",
    "ı",
    "ve",
    "defter",
    "ler",
    "i",
    "getir",
    "n",
    ",",
    "<UPCL>",
    "yo",
    "u",
    "<UPCL>",
    "tu",
    "be"
  ],
  "ids": [
    0,
    385,
    19936,
    19901,
    2001,
    19934,
    19935,
    159,
    19950,
    20022,
    0,
    643,
    19937,
    0,
    21941,
    21383
  ]
}
```

## Tokenization Rules

1. **Case Sensitivity**:
   - Words containing uppercase letters are split at each uppercase letter
   - Each part starting with an uppercase letter is preceded by `<UPCL>` token
   - All parts are converted to lowercase for processing

2. **Root Word Detection**:
   - Words are matched against the root dictionary
   - The longest possible match is used
   - Remaining characters are processed as suffixes or BPE tokens

3. **Suffix Handling**:
   - After finding a root word, remaining characters are matched against the suffix dictionary
   - Multiple suffixes can be detected in sequence
   - Longest possible match is used for each suffix

4. **BPE Fallback**:
   - If a part cannot be matched as a root or suffix, it is tokenized using byte-pair encoding
   - BPE ensures that any unknown text can still be tokenized

## Implementation Differences

While both implementations produce identical output, they have some technical differences:

1. **Performance**:
   - Rust implementation uses parallel processing via `rayon`
   - Rust version has more efficient memory management
   - Python version is more straightforward and easier to modify

2. **String Handling**:
   - Rust uses UTF-8 aware string operations
   - Python has native Unicode support

3. **Memory Usage**:
   - Rust implementation uses zero-cost abstractions
   - Python uses reference counting and garbage collection

## Examples

1. Simple word with suffix:
```bash
# Python
python3 turkish_tokenizer.py "kitabı"

# Rust
./target/release/turkish_tokenizer "kitabı"
```
Output:
```json
{
  "tokens": ["kitab", "ı"],
  "ids": [385, 19936]
}
```

2. Complex sentence with uppercase words:
```bash
# Python/Rust
"Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz"
```
Output:
```json
{
  "tokens": ["<UPCL>", "bir", "madde", "nin", "yan", "ma", "sı", "ile", "çık", "a", "n", "ve", "için", "de", "katı", "zerre", "ler", "le", "buğu", "bulun", "a", "n", "değişik", "renk", "ler", "de", "gaz"],
  "ids": [0, 1, 175, 19946, 59, 19954, 19952, 19888, 422, 19940, 19950, 19901, 19886, 19943, 926, 5976, 19934, 19947, 13592, 13, 19940, 19950, 273, 564, 19934, 19943, 965]
}
```

## Error Handling

Both implementations provide appropriate error messages for:
- Missing input text
- Missing dictionary files
- Invalid JSON in dictionary files
- Invalid UTF-8 sequences

## License

MIT

For more information about the project and other implementations, please refer to the [main documentation](../README.md). 