# Turkish Morphological Tokenizer

A high-performance morphological tokenizer for Turkish text, available in both Python and Rust implementations. The tokenizer handles root words, suffixes, and special cases like uppercase letters and punctuation.

## Features

- Morphological tokenization of Turkish text
- Root word and suffix detection
- Special handling of uppercase letters with `<uppercase>` tokens
- Byte-pair encoding (BPE) fallback for unknown tokens
- Support for Turkish characters and special characters (space, newline, tab)
- Thread-safe parallel processing (Rust implementation)
- Identical output format in both implementations

## Prerequisites

### For Python Implementation
- Python 3.6+
- Required dictionary files:
  - `kokler_v05.json`: Root words dictionary
  - `ekler_v05.json`: Suffixes dictionary
  - `bpe_v05.json`: BPE tokens dictionary

### For Rust Implementation
- Rust 1.50+
- Required dependencies (specified in `Cargo.toml`):
  - `serde` and `serde_json` for JSON handling
  - `regex` for text processing
  - `rayon` for parallel processing
  - `lazy_static` for efficient static resources
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
    "<uppercase>", "kitab", "ı", "<space>", "ve", "<space>", "defter", 
    "ler", "i", "<space>", "getir", "n", ",", "<newline>", 
    "<uppercase>", "yo", "u", "<uppercase>", "tu", "be", "<tab>"
  ],
  "ids": [
    0, 484, 22727, 1, 23790, 1, 2100, 22268, 22615, 1, 258, 
    22620, 22580, 2, 0, 742, 22627, 0, 24499, 23941, 3
  ]
}
```

## Special Tokens

```json
{
    "<uppercase>": 0,    // Uppercase letter marker
    "<space>": 1,       // Space character
    "<newline>": 2,     // Newline character
    "<tab>": 3,         // Tab character
    "<unknown>": 4      // Unknown token
}
```

## Token ID Ranges
- Special tokens: 0-4
- Root words: 5-20000
- Suffixes: 22268-22767
- BPE tokens: 20000+

## Tokenization Rules

1. **Case Sensitivity**:
   - Words containing uppercase letters are split at each uppercase letter
   - Each part starting with an uppercase letter is preceded by `<uppercase>` token
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

5. **Special Character Handling**:
   - Spaces are converted to `<space>` tokens
   - Newlines are converted to `<newline>` tokens
   - Tabs are converted to `<tab>` tokens
   - Other special characters are marked as `<unknown>`

## Implementation Differences

While both implementations produce identical output, they have some technical differences:

1. **Performance**:
   - Rust implementation uses parallel processing via `rayon`
   - Rust version has thread-safe data structures using `Arc`
   - Rust version uses zero-copy string operations
   - Python version is more straightforward and easier to modify

2. **String Handling**:
   - Rust uses efficient UTF-8 aware string operations
   - Rust implements proper lifetime management
   - Python has native Unicode support

3. **Memory Management**:
   - Rust uses zero-cost abstractions
   - Rust implementation uses string slices to reduce allocations
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
  "ids": [484, 22727]
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
  "tokens": ["<uppercase>", "bir", "madde", "nin", "yan", "ma", "sı", "ile", "çık", "a", "n", "ve", "için", "de", "katı", "zerre", "ler", "le", "buğu", "bulun", "a", "n", "değişik", "renk", "ler", "de", "gaz"],
  "ids": [0, 1, 175, 22727, 59, 22288, 22286, 19888, 422, 22274, 22284, 23790, 19886, 23779, 926, 5976, 22268, 23799, 13592, 13, 22274, 22284, 273, 564, 22268, 23779, 965]
}
```

## Error Handling

Both implementations provide appropriate error messages for:
- Missing input text
- Missing dictionary files
- Invalid JSON in dictionary files
- Invalid UTF-8 sequences
- Thread safety violations (Rust only)

## Development Workflow

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd turkish_tokenizer
```

2. Install Rust dependencies:
```bash
cargo build
```

3. Run tests:
```bash
cargo test
```

4. Format code:
```bash
cargo fmt
```

5. Run linter:
```bash
cargo clippy
```

### Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Ensure all tests pass:
```bash
cargo test
```

4. Format code and check for linting issues:
```bash
cargo fmt
cargo clippy
```

5. Commit your changes:
```bash
git add .
git commit -m "feat: description of your changes"
```

### Pull Request Process

1. Push your changes:
```bash
git push origin feature/your-feature-name
```

2. Create a Pull Request on GitHub
3. Wait for code review
4. Make requested changes if any
5. Once approved, your PR will be merged

## Performance Considerations

When making changes, keep in mind:

1. **Memory Usage**:
   - Use string slices (`&str`) instead of `String` where possible
   - Avoid unnecessary cloning
   - Use zero-copy operations when feasible

2. **Thread Safety**:
   - Ensure shared data structures are wrapped in `Arc`
   - Use proper synchronization primitives
   - Test with parallel processing enabled

3. **Error Handling**:
   - Provide descriptive error messages
   - Handle all potential error cases
   - Use proper error propagation

4. **UTF-8 Handling**:
   - Always use proper UTF-8 string operations
   - Test with various Turkish characters
   - Handle invalid UTF-8 sequences gracefully

## License

MIT

For more information about the project and other implementations, please refer to the [main documentation](../README.md). 