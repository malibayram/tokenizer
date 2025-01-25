# Turkish Morphological Tokenizer

A Rust-based morphological tokenizer for Turkish text that handles root words, suffixes, and special cases like uppercase letters and punctuation.

## Features

- Morphological tokenization of Turkish text
- Root word and suffix detection
- Special handling of uppercase letters with `<UPCL>` tokens
- Byte-pair encoding (BPE) fallback for unknown tokens
- Support for Turkish characters
- JSON output format

## Prerequisites

- Rust (latest stable version)
- Required dictionary files:
  - `kokler_v04.json`: Root words dictionary
  - `ekler_v04.json`: Suffixes dictionary
  - `bpe_v02.json`: BPE tokens dictionary

## Installation

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

### Command Line

The tokenizer accepts input text as a command-line argument:

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

The tokenizer outputs JSON with two arrays:
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
    "t",
    "u",
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
    20068,
    20069,
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
   - If no exact match is found, the word is progressively shortened from the end

3. **Suffix Handling**:
   - After finding a root word, remaining characters are matched against the suffix dictionary
   - Multiple suffixes can be detected in sequence

4. **BPE Fallback**:
   - If a part cannot be matched as a root or suffix, it is tokenized using byte-pair encoding
   - BPE ensures that any unknown text can still be tokenized

## Examples

1. Simple word with suffix:
```bash
./target/release/turkish_tokenizer "kitabı"
```
Output:
```json
{
  "tokens": ["kitab", "ı"],
  "ids": [385, 19936]
}
```

2. Word with uppercase letters:
```bash
./target/release/turkish_tokenizer "YouTube"
```
Output:
```json
{
  "tokens": ["<UPCL>", "yo", "u", "<UPCL>", "t", "u", "be"],
  "ids": [0, 643, 19937, 0, 20068, 20069, 21383]
}
```

## Error Handling

If you run the tokenizer without any input text, it will display a usage message:
```bash
./target/release/turkish_tokenizer
Usage: ./target/release/turkish_tokenizer <input_text>
```

## License

MIT

For more information about the project and other implementations, please refer to the [main documentation](../README.md). 