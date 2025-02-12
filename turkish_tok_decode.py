import json 
import os

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

#ÜST TARAF ZATEN SİLİNECEK

vowels = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]
consonants = ["b", "c", "ç", "d", "f", "g", "ğ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "ş", "t", "v", "y", "z"]
back_vowels = ["a", "ı", "o", "u"]
front_vowels = ["e", "i", "ö", "ü"]
hard_consonants = ["ç", "f", "h", "k", "p", "s", "ş", "t"]
consonant_softening_dict = {
    'ç': 'c',   
    'p': 'b',
    't': 'd',
    'k': 'ğ',
}

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

# def is_back_vowel(word):
#     for i in range(len(word)):
#         if word[i] in back_vowels:
#             return True
#     return False

def vowel_thinner(word):
    for i in range(len(word)):
        if word[i] in "a":
            word[i] = "e"
    return word

def choose_correct_version(cur_token: int, next_token: str, prev_token: str, dict):
    tokens = dict[cur_token] 
    i = 0
    while len(tokens) > 1:
        if cur_token <= 2267:
            #yumuşama 
            for i in range(len(tokens)):
                if tokens[i][-1] in hard_consonants and consonant_softening(tokens[i]) in tokens:
                    if not next_token[0] in vowels:
                        tokens.pop(consonant_softening(tokens[i]))
                    else:
                        tokens.pop(tokens[i])
            #düşme 
            for i in range(len(tokens)):
                if tokens[i][-1] in consonants and vowel_reduction(tokens[i]) in tokens:
                    if not next_token[0] in vowels:
                        tokens.pop(vowel_reduction(tokens[i]))
                    else:
                        tokens.pop(tokens[i])
            #daralma
            for i in range(len(tokens)):
                if tokens[i][-1] in ["a", "e"] and narrow_vowel(tokens[i]) in tokens:
                    if not next_token[0].startswith("yor"):
                        tokens.pop(narrow_vowel(tokens[i]))
                    else:
                        tokens.pop(tokens[i])
        else:
            #den ten
            for i in range(len(tokens)):
                if tokens[i][0] == "d" and suffix_hardening(tokens[i]) in tokens:
                    if not prev_token[-1] in hard_consonants:
                        tokens.pop(suffix_hardening(tokens[i]))
                    else:
                        tokens.pop(tokens[i])
            #ler lar 
            for i in range(len(tokens)):
                if first_vowel(tokens[i]) in back_vowels and vowel_thinner(tokens[i]) in tokens:
                    if not last_vowel(prev_token) in back_vowels:
                        tokens.pop(tokens[i])
                    else:
                        tokens.pop(vowel_thinner(tokens[i]))
    return tokens[0]
                        
            
    

def decode_text(list, dict):
    prev_token
    result = ""
    for i in range(len(list)):
        cur_token = list(i)
        next_token = list(i+1)
        
        if(dict[cur_token] == "[UNKOWN]"):
            prev_token = cur_token
            continue
        if(len(dict[cur_token]) == 1):
            if(dict[prev_token] == "[UPPERCASE]"):
                result += dict[cur_token].upper()
                prev_token = cur_token
                continue
            result += dict[cur_token]
            prev_token = cur_token
            continue
        elif(len(dict[cur_token]) > 1):
            result += choose_correct_version(cur_token, dict[next_token], dict[prev_token], dict)
            prev_token = cur_token
            continue
        else:
            continue
    return result

ids = input("Enter the ids: ")

    
    
    