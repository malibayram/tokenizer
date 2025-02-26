import json
import re
import os
from typing import List, Dict, Tuple

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load JSON files into memory
def load_json(file_path: str) -> Dict[str, int]:
    full_path = os.path.join(CURRENT_DIR, file_path)
    with open(full_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load roots, suffixes, and BPE tokens
roots = load_json("kokler_v07.json")
suffixes = load_json("ekler_v05.json")
bpe_tokens = load_json("bpe_v05.json")

reverse_dict = {}

for key, value in roots.items():
    if value not in reverse_dict:
        reverse_dict[value] = []
    reverse_dict[value].append(key)
    
for key, value in suffixes.items():
    if value not in reverse_dict:
        reverse_dict[value] = []
    reverse_dict[value].append(key)
    
for key, value in bpe_tokens.items():
    if value not in reverse_dict:
        reverse_dict[value] = []
    reverse_dict[value].append(key)
    

# Special token IDs
SPECIAL_TOKENS = {
    "<space>": 1,
    "<newline>": 2,
    "<tab>": 3,
    "<unknown>": 4,
    "<uppercase>": 0
}

# Tokenize the input text
def tokenize(text: str) -> Dict[str, List]:
    tokens = []
    ids = []

    # Process each character for special whitespace tokens
    i = 0
    while i < len(text):
        char = text[i]
        if char == ' ':
            tokens.append("<space>")
            ids.append(SPECIAL_TOKENS["<space>"])
        elif char == '\n':
            tokens.append("<newline>")
            ids.append(SPECIAL_TOKENS["<newline>"])
        elif char == '\t':
            tokens.append("<tab>")
            ids.append(SPECIAL_TOKENS["<tab>"])
        elif char.isalnum() or char in '.,!?;':
            # Collect the word or punctuation
            word_start = i
            while i < len(text) and (text[i].isalnum() or text[i] in '.,!?;'):
                i += 1
            word = text[word_start:i]
            i -= 1  # Adjust for the outer loop increment
            
            if any(c.isupper() for c in word):
                # Split by uppercase letters and process each part
                parts = re.split(r'([A-Z][^A-Z]*)', word)
                for part in parts:
                    if part:
                        if part[0].isupper():
                            tokens.append("<uppercase>")
                            ids.append(SPECIAL_TOKENS["<uppercase>"])
                            process_word(part.lower(), tokens, ids)
                        else:
                            process_word(part, tokens, ids)
            else:
                process_word(word, tokens, ids)
        i += 1

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
        # Try BPE tokenization
        bpe_success = process_bpe(word, tokens, ids)
        if not bpe_success:
            # If no matches found, mark as unknown
            tokens.append("<unknown>")
            ids.append(SPECIAL_TOKENS["<unknown>"])

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
            # Try BPE tokenization
            bpe_success = process_bpe(remainder, tokens, ids)
            if not bpe_success:
                # If no matches found, mark as unknown
                tokens.append("<unknown>")
                ids.append(SPECIAL_TOKENS["<unknown>"])

# Match a suffix from the suffixes dictionary
def match_suffix(word: str) -> Tuple[str, int]:
    for i in range(len(word), 0, -1):
        if word[:i] in suffixes:
            return word[:i], suffixes[word[:i]]
    return None, None

# Process a word using BPE
def process_bpe(word: str, tokens: List[str], ids: List[int]) -> bool:
    i = 0
    found_any = False
    while i < len(word):
        found_match = False
        for j in range(len(word), i, -1):
            if word[i:j] in bpe_tokens:
                tokens.append(word[i:j])
                ids.append(bpe_tokens[word[i:j]])
                i = j
                found_match = True
                found_any = True
                break
        if not found_match:
            i += 1
    return found_any

# Example execution
if __name__ == "__main__":
    input_texts = [
        "Kitabı ve defterleri getirn,\nYouTube\t",
        "Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz"
    ]
    for input_text in input_texts:
        print(input_text)
        result = tokenize(input_text)
        print(result)

    """ Example outputs:
    Kitabı ve defterleri getirn,
    YouTube	
    {'tokens': ['<uppercase>', 'kitab', 'ı', '<space>', 've', '<space>', 'defter', 'ler', 'i', '<space>', 'getir', 'n', ',', '<newline>', '<uppercase>', 'you', '<uppercase>', 'tube', '<tab>'], 'ids': [0, 385, 22270, 1, 19901, 1, 2001, 22268, 22269, 1, 159, 22284, 20022, 2, 0, 643, 0, 21941, 3]}
    
    Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz
    {'tokens': ['<uppercase>', 'bir', '<space>', 'madde', 'nin', '<space>', 'yan', 'ma', 'sı', '<space>', 'ile', '<space>', 'çık', 'a', 'n', '<space>', 've', '<space>', 'için', 'de', '<space>', 'katı', '<space>', 'zerre', 'ler', 'le', '<space>', 'buğu', '<space>', 'bulun', 'a', 'n', '<space>', 'değişik', '<space>', 'renk', 'ler', 'de', '<space>', 'gaz'], 'ids': [0, 1, 1, 175, 22280, 1, 59, 22288, 22286, 1, 19888, 1, 422, 22274, 22284, 1, 19901, 1, 19886, 22277, 1, 926, 1, 5976, 22268, 22281, 1, 13592, 1, 13, 22274, 22284, 1, 273, 1, 564, 22268, 22277, 1, 965]} 
    """
    
# ------ DECODE PART ------


# These are some useful lists and dictionaries that are used in the decoding part
vowels = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]
consonants = ["b", "c", "ç", "d", "f", "g", "ğ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "ş", "t", "v", "y", "z"]
back_vowels = ["a", "ı", "o", "u"]
front_vowels = ["e", "i", "ö", "ü"]
suffix_group = ["i", "ı", "u", "ü"]
hard_consonants = ["ç", "f", "h", "k", "p", "s", "ş", "t"]
consonant_softening_dict = {
    'ç': 'c',   
    'p': 'b',
    't': 'd',
    'k': 'ğ',
}

# These are the helper functions that are used in the decoding part
def consonant_softening(word):
    if word[-1] in consonant_softening_dict:
        return word[:-1] + consonant_softening_dict[word[-1]]
    else:
        return word
    
def suffix_hardening(suffix):
    if suffix[0] == "d":
        return "t" + suffix[1:]
    return suffix

def vowel_reduction(word):  
    if word[-2] in vowels:
        word = word[:-2] + word[-1]
    return word

def narrow_vowel(word):
    if word[-1] in ["a"]:
        if (word[:-1] + "ı") in roots:
            return word[:-1] + "ı"
        elif (word[:-1] + "u") in roots:
            return word[:-1] + "u"
    elif word[-1] in ["e"]:
        if (word[:-1] + "i") in roots:
            return word[:-1] + "i"
        elif (word[:-1] + "ü") in roots:
            return word[:-1] + "ü"

def first_vowel(word):
    for i in range(len(word)):
        if word[i] in vowels:
            return i
    return -1

def last_vowel(word):
    for i in range(len(word)-1, -1, -1):
        if word[i] in vowels:
            return i
    return -1

def vowel_variator(word):
    variations = [word]
    variations.append(word[:first_vowel(word)] + "i" + word[first_vowel(word)+1:])
    variations.append(word[:first_vowel(word)] + "ı" + word[first_vowel(word)+1:])
    variations.append(word[:first_vowel(word)] + "u" + word[first_vowel(word)+1:])
    variations.append(word[:first_vowel(word)] + "ü" + word[first_vowel(word)+1:])
    variations = list(set(variations))
    return variations
    

def vowel_thinner(word):
    for i in range(len(word)):
        if word[i] in "a":
            word[i] = "e"
    return word


'''
This function takes a list of tokens and returns the correct version of the token.
The function first checks if the token is a root or a suffix by looking to index of it.
If the token is root than it checks "if there is a variation of that token and chooses the correct version.
If the token is suffix than it checks if there is a variation of that suffix and chooses the correct version.
'''
def choose_correct_version(cur_token: list, next_token: list, prev_token: list, cur_token_id: int) -> str:
    # If token is a root
    if cur_token_id <= 2267:
        # If there is softened form of token 
        for token in cur_token.copy():
            if not token:
                continue
            if token[-1] in hard_consonants and consonant_softening(token) in cur_token:
                if not next_token[0][0] in vowels:
                    cur_token.remove(consonant_softening(token))
                else:
                    cur_token.remove(token)
        # If there is a reduced form of token
        for i in range(len(cur_token)):
            if not token:
                continue
            if token[-1] in consonants and vowel_reduction(token) in cur_token:
                if not next_token[0][0] in vowels:
                    cur_token.remove(vowel_reduction(token))
                else:
                    cur_token.remove(token)
        # If there is a narrow vowel form of token
        for i in range(len(cur_token)):
            if not token:
                continue
            if token[-1] in ["a", "e"] and narrow_vowel(token) in cur_token:
                if not next_token[0].startswith("yor"):
                    cur_token.remove(narrow_vowel(token))
                else:
                    cur_token.remove(token)
    # If token is a suffix
    else:
        # "e-a" variation of suffixes
        for i in range(len(cur_token)):
            print("e-a")
            fv_index = first_vowel(token)
            if fv_index is None or fv_index == -1 or fv_index >= len(token):
                continue
            if token[first_vowel(token)] in back_vowels and vowel_thinner(token) in cur_token:
                if not prev_token[last_vowel(prev_token)] in back_vowels:
                    cur_token.remove(token)
                else:
                    cur_token.remove(vowel_thinner(token))
        # "ı-i-u-ü" variation of suffixes
        for i in range(len(cur_token)):
            if token[first_vowel(token)] in suffix_group and any(variation in vowel_variator(token) for variation in cur_token):
                # Below if-else if block checks that which variation of the suffix is correct according to the previous token
                if prev_token[last_vowel(prev_token)] == "a" or prev_token[last_vowel(prev_token)] == "ı":
                    for variation in vowel_variator(token):
                        if variation in cur_token and variation[first_vowel(variation)] != "ı":
                            cur_token.remove(variation)
                elif prev_token[last_vowel(prev_token)] == "e" or prev_token[last_vowel(prev_token)] == "i":
                    for variation in vowel_variator(token):
                        if variation in cur_token and variation[first_vowel(variation)] != "i":
                            cur_token.remove(variation)
                elif prev_token[last_vowel(prev_token)] == "o" or prev_token[last_vowel(prev_token)] == "u":
                    for variation in vowel_variator(token):
                        if variation in cur_token and variation[first_vowel(variation)] != "u":
                            cur_token.remove(variation)
                        if variation in cur_token and variation[first_vowel(variation)] != "u":
                            cur_token.remove(variation)
                elif prev_token[last_vowel(prev_token)] == "ö" or prev_token[last_vowel(prev_token)] == "ü":
                    for variation in vowel_variator(token):
                        if variation in cur_token and variation[first_vowel(variation)] != "ü":
                            cur_token.remove(variation)
        # "d-t" variation of suffixes
        for i in range(len(cur_token)):
            if not token:
                continue
            if token[0] == "d" and suffix_hardening(token) in cur_token:
                if not prev_token[-1] in hard_consonants:
                    cur_token.remove(suffix_hardening(token))
                else:
                    cur_token.remove(token)
    if not cur_token:
        return ""
    else:
        return cur_token[0]

                        
#def choose_between_two(cur_tokenn):
    
    
'''
This is the function that decodes the tokenized text into a readable text.
The function takes id's a list and then checks if that id has a corresponding token in the reverse_dict.
If the corresponding value has one variation of token then it directly adds it to the result.
If the corresponding value has more than one variation of token then it calls choose_correct_version function to add correct form to result.
'''
def decode_text(list):
    prev_token = reverse_dict[list[0]]
    result = ""
    # Iterating through the list
    for i in range(len(list)):
        cur_token = reverse_dict[list[i]]
        next_token = []
        print(cur_token)
        if i != len(list) - 1: 
            next_token = reverse_dict[list[i + 1]]
        
        if not cur_token:
            continue
        # Checking if the token is UNKOWN
        if (cur_token[0] == "[UNKOWN]"):
            prev_token = cur_token
            continue
        # Checking if the token is SPACE
        if (cur_token[0] == "<space>"):
            result += " "
            prev_token = cur_token
            continue
        # Checking if the token is NEWLINE
        if (cur_token[0] == "<newline>"):
            result += "\n"
            prev_token = cur_token
            continue
        # Checking if the token is TAB
        if (cur_token[0] == "<tab>"):
            result += "\t"
            prev_token = cur_token
            continue
        # Checking if the token is UPPERCASE
        if (cur_token[0] == "<uppercase>"):
            prev_token = cur_token[0]
            continue
        if (prev_token == "<uppercase>"):
            result += cur_token[0].capitalize()
            prev_token = cur_token[0]
            continue
        # Checking if the id has more than one corresponding token
        if (len(cur_token) == 1):
            result += cur_token[0]
            prev_token = cur_token
            continue
        # If there are more than one token corresponding to the id we call choose_correct_version function to add correct form to result
        elif (len(cur_token) > 1):
            result += choose_correct_version(cur_token, next_token, prev_token, i)
            prev_token = cur_token
            continue
        else:
            continue
    return result
