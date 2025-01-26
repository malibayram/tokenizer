import json
import re
from typing import List, Dict, Tuple

# Load JSON files into memory
def load_json(file_path: str) -> Dict[str, int]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load roots, suffixes, and BPE tokens
roots = load_json("kokler_v04.json")
suffixes = load_json("ekler_v04.json")
bpe_tokens = load_json("bpe_v02.json")

# Tokenize the input text
def tokenize(text: str) -> Dict[str, List]:
    tokens = []
    ids = []

    # Split the text into words and punctuation
    words = re.findall(r'[\w]+|[.,!?;]', text)

    for word in words:
        if any(c.isupper() for c in word):
            # Split by uppercase letters and process each part
            parts = re.split(r'([A-Z][^A-Z]*)', word)
            for part in parts:
                if part:
                    if part[0].isupper():
                        tokens.append("<UPCL>")
                        ids.append(roots["<UPCL>"])
                        process_word(part.lower(), tokens, ids)
                    else:
                        process_word(part, tokens, ids)
        else:
            process_word(word, tokens, ids)

    return {"tokens": tokens, "ids": ids}

# Process a single word
def process_word(word: str, tokens: List[str], ids: List[int]):
    # Check if the word matches a root
    root, root_id, remainder = match_root(word)

    if root:
        tokens.append(root)
        ids.append(root_id)
        if remainder:
            process_remainder(remainder, tokens, ids)
    else:
        # Fallback to BPE if no root match is found
        process_bpe(word, tokens, ids)

# Match a root from the roots dictionary
def match_root(word: str) -> Tuple[str, int, str]:
    for i in range(len(word), 1, -1):
        if word[:i] in roots:
            return word[:i], roots[word[:i]], word[i:]
    return None, None, word

# Process the remainder of the word
def process_remainder(remainder: str, tokens: List[str], ids: List[int]):
    # Check if the remainder matches a suffix
    suffix, suffix_id = match_suffix(remainder)

    if suffix:
        tokens.append(suffix)
        ids.append(suffix_id)
        remainder = remainder[len(suffix):]

        if remainder:
            process_remainder(remainder, tokens, ids)
    else:
        # Check if the remainder matches another root
        root, root_id, remainder = match_root(remainder)
        if root:
            tokens.append(root)
            ids.append(root_id)
            if remainder:
                process_remainder(remainder, tokens, ids)
        else:
            # Fallback to BPE if no match
            process_bpe(remainder, tokens, ids)

# Match a suffix from the suffixes dictionary
def match_suffix(word: str) -> Tuple[str, int]:
    for i in range(len(word), 0, -1):
        if word[:i] in suffixes:
            return word[:i], suffixes[word[:i]]
    return None, None

# Process a word using BPE
def process_bpe(word: str, tokens: List[str], ids: List[int]):
    i = 0
    while i < len(word):
        for j in range(len(word), i, -1):
            if word[i:j] in bpe_tokens:
                tokens.append(word[i:j])
                ids.append(bpe_tokens[word[i:j]])
                i = j - 1
                break
        i += 1

# Example execution
if __name__ == "__main__":
    input_texts = [
      "Kitabı ve defterleri getirn, YouTube",
      "Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz"
    ]
    for input_text in input_texts:
        print(input_text)
        result = tokenize(input_text)
        print(result)

    """ Kitabı ve defterleri getirn, YouTube
    {'tokens': ['<UPCL>', 'kitab', 'ı', 've', 'defter', 'ler', 'i', 'getir', 'n', ',', '<UPCL>', 'yo', 'u', '<UPCL>', 'tu', 'be'], 'ids': [0, 385, 19936, 19901, 2001, 19934, 19935, 159, 19950, 20022, 0, 643, 19937, 0, 21941, 21383]}
    Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz
    {'tokens': ['<UPCL>', 'bir', 'madde', 'nin', 'yan', 'ma', 'sı', 'ile', 'çık', 'a', 'n', 've', 'için', 'de', 'katı', 'zerre', 'ler', 'le', 'buğu', 'bulun', 'a', 'n', 'değişik', 'renk', 'ler', 'de', 'gaz'], 'ids': [0, 1, 175, 19946, 59, 19954, 19952, 19888, 422, 19940, 19950, 19901, 19886, 19943, 926, 5976, 19934, 19947, 13592, 13, 19940, 19950, 273, 564, 19934, 19943, 965]} 
    """
