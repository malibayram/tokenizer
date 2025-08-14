import json
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TokenType(Enum):
    ROOT = "ROOT"
    SUFFIX = "SUFFIX"
    BPE = "BPE"

class TRTokenizer:
    def __init__(self):
        with open("v09_kokler_new.json", "r") as f:
            roots = json.load(f)
        with open("v09_ekler.json", "r") as f:
            suffixes = json.load(f)
        with open("v09_bpe_tokens.json", "r") as f:
            bpe_tokens = json.load(f)
        self.reverse_dict = {}

        for key, value in roots.items():
            if value not in self.reverse_dict:
                self.reverse_dict[value] = []
            self.reverse_dict[value].append(key)
        for key, value in suffixes.items():
            if value not in self.reverse_dict:
                self.reverse_dict[value] = []
            self.reverse_dict[value].append(key)
        for key, value in bpe_tokens.items():
            if value not in self.reverse_dict:
                self.reverse_dict[value] = []
            self.reverse_dict[value].append(key)

        self.roots = roots
        self.suffixes = suffixes
        self.bpe_tokens = bpe_tokens
        self.max_root_len = max(len(k) for k in roots) if roots else 0
        self.max_suffix_len = max(len(k) for k in suffixes) if suffixes else 0
        self.max_bpe_len = max(len(k) for k in bpe_tokens) if bpe_tokens else 0
        
        self.uppercase_marker = {"token": "<uppercase>", "id": 0, "type": TokenType.ROOT}
        self.space_marker = {"token": "<space>", "id": 1, "type": TokenType.ROOT}
        self.unknown_marker = {"token": "<unknown>", "id": 4, "type": TokenType.ROOT}

    def _tokenize_word(self, word: str) -> Tuple[List[dict], List[int]]:
        uppercase_indices = [i for i, c in enumerate(word) if c.isupper()]
        result = []
        
        segments = self._camel_split_with_positions(word)
        
        for seg, orig_pos in segments:
            if orig_pos < len(word) and word[orig_pos].isupper():
                result.append(self.uppercase_marker)
            
            s = seg
            pos = 0
            
            while pos < len(s):
                substr = s[pos:]
                
                rid, rtok = self._longest_prefix_lookup(substr, self.roots, self.max_root_len)
                if rid is not None:
                    result.append({"token": rtok, "id": rid, "type": TokenType.ROOT})
                    pos += len(rtok)
                    continue
                
                sid, stok = self._longest_prefix_lookup(substr, self.suffixes, self.max_suffix_len)
                if sid is not None:
                    result.append({"token": stok, "id": sid, "type": TokenType.SUFFIX})
                    pos += len(stok)
                    continue
                
                bid, btok = self._longest_prefix_lookup(substr, self.bpe_tokens, self.max_bpe_len)
                if bid is not None:
                    result.append({"token": btok, "id": bid, "type": TokenType.BPE})
                    pos += len(btok)
                    continue
                
                result.append(self.unknown_marker)
                pos += 1
        
        return result, uppercase_indices

    def tokenize_text(self, text: str) -> Tuple[List[dict], List[int]]:
        final_tokens = []
        uppercase_indices = [i for i, c in enumerate(text) if c.isupper()]
        
        parts = text.split(" ")
        for idx, part in enumerate(parts):
            if part.strip():
                tokens, _ = self._tokenize_word(part)
                final_tokens.extend(tokens)
            if idx < len(parts) - 1:
                final_tokens.append(self.space_marker)
        
        return final_tokens, uppercase_indices
    
    def encode(self, text: str) -> List[int]:
        tokens, _ = self.tokenize_text(text)
        return [t["id"] for t in tokens]
    
    def tokenize(self, text: str) -> List[str]:
        tokens, _ = self.tokenize_text(text)
        return [t["token"] for t in tokens]
    
    def _longest_prefix_lookup(self, s: str, table: Dict[str, int], max_len: int = None) -> Tuple[Optional[int], str]:
        end = min(len(s), max_len) if max_len else len(s)
        for i in range(end, 0, -1):
            cand = s[:i]
            if cand in table:
                return table[cand], cand
        return None, ""

    def _camel_split_with_positions(self, word: str) -> List[Tuple[str, int]]:
        if not word:
            return []
        
        parts = []
        start = 0
        
        for i in range(1, len(word)):
            if word[i].isupper():
                if start < i:
                    parts.append((word[start:i].lower(), start))
                start = i
        
        if start < len(word):
            parts.append((word[start:].lower(), start))
        
        return parts if parts else [(word.lower(), 0)]

    def _starts_with_vowel(self, word: str) -> bool:
        return word[0] in "aeıioöuü"

    def _ends_with_vowel(self, word: str) -> bool:
        return word[len(word) - 1] in "aeıioöuü"

    def _ends_with_any(word: str, charset: str, k: int = 3) -> bool:
        # Safely check last k chars; slicing handles short words gracefully
        return any(c in charset for c in word[-k:]) if word else False

    def _ends_with_ince(self, word: str) -> bool:
        return self._ends_with_any(word, "eiöü")

    def _ends_with_ai(self, word: str) -> bool:
        return self._ends_with_any(word, "aı")

    def _ends_with_ei(self, word: str) -> bool:
        return self._ends_with_any(word, "ei")

    def _ends_with_ou(self, word: str) -> bool:
        return self._ends_with_any(word, "ou")
    
    def _ends_with_sert_unsuz(self, word: str) -> bool:
        return word[len(word) - 1] in "fstkçşhp"

    def _select_correct_suffix(self, i: int, ids: List[int]) -> str:
        suffixes = self.reverse_dict[ids[i]]
        if ids[i] < 20013 and i > 0:
            prev_token = self.reverse_dict[ids[i - 1]][0]
            if (prev_token == " " or prev_token == "\n" or prev_token == "\t") and i > 1:
                prev_token = self.reverse_dict[ids[i - 2]][0]
            if self._ends_with_ince(prev_token):
                return suffixes[1]
            return suffixes[0]
        elif ids[i] < 20023 and i > 0: # nın, nin, nun, nün
            prev_token = self.reverse_dict[ids[i - 1]][0]
            if self._ends_with_ai(prev_token):
                return suffixes[0]
            elif self._ends_with_ei(prev_token):
                return suffixes[1]
            elif self._ends_with_ou(prev_token):
                return suffixes[2]
            return suffixes[3]
        elif ids[i] == 20023 and i > 0: # la, le, yla, yle
            prev_token = self.reverse_dict[ids[i - 1]][0]
            if self._ends_with_vowel(prev_token):
                if self._ends_with_ince(prev_token):
                    return suffixes[3]
                return suffixes[2]
            elif self._ends_with_ince(prev_token):
                return suffixes[1]
            return suffixes[0]
        elif ids[i] <= 20025  and i > 0: # da, de, ta, te, dan, den, tan, ten
            prev_token = self.reverse_dict[ids[i - 1]][0]
            if (prev_token == " " or prev_token == "\n" or prev_token == "\t") and i > 1:
                prev_token = self.reverse_dict[ids[i - 2]][0]
            if self._ends_with_sert_unsuz(prev_token):
                if self._ends_with_ince(prev_token):
                    return suffixes[3]
                return suffixes[2]
            if self._ends_with_ince(prev_token):
                return suffixes[1]
            return suffixes[0]
        elif ids[i] > 20025 and ids[i] < 20029: # dı, di, du, dü, tı, ti, tu, tü, cı, ci, cu, cü, çı, çi, çu, çü
            prev_token = self.reverse_dict[ids[i - 1]][0]
            if self._ends_with_sert_unsuz(prev_token):
                if self._ends_with_ai(prev_token):
                    return suffixes[4]
                elif self._ends_with_ei(prev_token):
                    return suffixes[5]
                elif self._ends_with_ou(prev_token):
                    return suffixes[6]
                return suffixes[7]
            else:
                if self._ends_with_ai(prev_token):
                    return suffixes[0]
                elif self._ends_with_ei(prev_token):
                    return suffixes[1]
                elif self._ends_with_ou(prev_token):
                    return suffixes[2]
                return suffixes[3]
        elif ids[i] == 20029: # lık, lik, luk, lük, lığ, liğ, luğ, lüğ
            prev_token = self.reverse_dict[ids[i - 1]][0]
            next_token = self.reverse_dict[ids[i + 1]][0]
            if self._starts_with_vowel(next_token):
                if self._ends_with_ai(prev_token):
                    return suffixes[4]
                elif self._ends_with_ei(prev_token):
                    return suffixes[5]
                elif self._ends_with_ou(prev_token):
                    return suffixes[6]
                return suffixes[7]
            else:
                if self._ends_with_ai(prev_token):
                    return suffixes[0]
                elif self._ends_with_ei(prev_token):
                    return suffixes[1]
                elif self._ends_with_ou(prev_token):
                    return suffixes[2]
                return suffixes[3]
        elif ids[i] == 20030: # cık, cik, cuk, cük, çık, çik, çuk, çük, cığ, ciğ, cuğ, cüğ, çığ, çiğ, çuğ, çüğ
            prev_token = self.reverse_dict[ids[i - 1]][0]
            next_token = self.reverse_dict[ids[i + 1]][0]
            if self._starts_with_vowel(next_token):
                if self._ends_with_sert_unsuz(prev_token):
                    if self._ends_with_ai(prev_token):
                        return suffixes[12]
                    elif self._ends_with_ei(prev_token):
                        return suffixes[13]
                    elif self._ends_with_ou(prev_token):
                        return suffixes[14]
                    return suffixes[15]
                else:
                    if self._ends_with_ai(prev_token):
                        return suffixes[8]
                    elif self._ends_with_ei(prev_token):
                        return suffixes[9]
                    elif self._ends_with_ou(prev_token):
                        return suffixes[10]
                    return suffixes[11]
            else:
                if self._ends_with_sert_unsuz(prev_token):
                    if self._ends_with_ai(prev_token):
                        return suffixes[4]
                    elif self._ends_with_ei(prev_token):
                        return suffixes[5]
                    elif self._ends_with_ou(prev_token):
                        return suffixes[6]
                    return suffixes[7]
                else:
                    if self._ends_with_ai(prev_token):
                        return suffixes[0]
                    elif self._ends_with_ei(prev_token):
                        return suffixes[1]
                    elif self._ends_with_ou(prev_token):
                        return suffixes[2]
                    return suffixes[3]
        elif ids[i] == 20031: # mak, mek, may, mey
            prev_token = self.reverse_dict[ids[i - 1]][0]
            next_token = self.reverse_dict[ids[i + 1]][0]
            if self._starts_with_vowel(next_token):
                if self._ends_with_ince(prev_token):
                    return suffixes[3]
                return suffixes[2]
            else:
                if self._ends_with_ince(prev_token):
                    return suffixes[1]
                return suffixes[0]
        elif ids[i] == 20032: # acak, ecek, acağ, eceğ, yacak, yecek, yacağ, yeceğ
            prev_token = self.reverse_dict[ids[i - 1]][0]
            next_token = self.reverse_dict[ids[i + 1]][0]
            if self._starts_with_vowel(next_token):
                if self._ends_with_vowel(prev_token):
                    if self._ends_with_ince(prev_token):
                        return suffixes[7]
                    return suffixes[6]
                else:
                    if self._ends_with_ince(prev_token):
                        return suffixes[3]
                    return suffixes[2]
            else:
                if self._ends_with_vowel(prev_token):
                    if self._ends_with_ince(prev_token):
                        return suffixes[5]
                    return suffixes[4]
                else:
                    if self._ends_with_ince(prev_token):
                        return suffixes[1]
                    return suffixes[0]
        else:
            return suffixes[0]

    """ 
    {
        'üçlü, yumuşama ve ünsüz düşmesi': 100,
        'ünlü-düşme': 110,
        'yumuşama': 198,
        'kalın-genişleme': 2080,
        'ince-genişleme': 2223,
        'yanlış eklenmiş': 2315
    }
    """
    def _select_correct_root(self, i: int, ids: List[int]) -> str:
        if ids[i] >= 100 and ids[i] < 2080 and i < len(ids) - 2:
            next_token = self.reverse_dict[ids[i + 1]][0]
            if self._starts_with_vowel(next_token):
                return self.reverse_dict[ids[i]][1]
            elif ids[i] <= 110 and ids[i + 1] == 20034: # cık, cik, ...
                return self.reverse_dict[ids[i]][2]
            else:
                return self.reverse_dict[ids[i]][0]
        elif ids[i] >= 2080 and ids[i] < 2315 and i < len(ids) - 2: # kalın-genişleme, ince-genişleme, ...
            next_token = self.reverse_dict[ids[i + 1]][0]
            if ids[i + 1] == 20021: # yor
                return self.reverse_dict[ids[i]][1]
            else:
                return self.reverse_dict[ids[i]][0]
        else:
            return self.reverse_dict[ids[i]][0]


    def decode(self, ids: List[int]) -> str:
        text = ""
        i = 0
        while i < len(ids):
            if ids[i] == 0 and i < len(ids) - 1: # uppercase
                # get the token is uppercase then uppercase the next token's first letter
                token = self.reverse_dict[ids[i + 1]][0]
                text += token.capitalize()
                i += 1
            elif ids[i] == 1: # space
                text += " "
            elif ids[i] == 2: # newline
                text += "\n"
            elif ids[i] == 3: # tab
                text += "\t"
            elif ids[i] == 4: # unknown
                text += "▁u▁"
            elif ids[i] in self.reverse_dict:
                tokens = self.reverse_dict[ids[i]]
                if len(tokens) > 1 and i > 0:
                    if ids[i] < 20000: # if the token is a root
                        text += self._select_correct_root(i, ids)
                    else: # token is a suffix since bpe tokens are single tokens
                        text += self._select_correct_suffix(i, ids)
                else:
                    text += tokens[0]
            else:
                text += "▁"
            i += 1
        return text
