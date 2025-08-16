from typing import List


class TRDecoder:
    # Define vowel sets as class constants for better performance
    ALL_VOWELS = "aeıioöuü"
    INCE_VOWELS = "eiöü"  # Front vowels
    AI_VOWELS = "aı"      # Back unrounded
    EI_VOWELS = "ei"      # Front unrounded  
    OU_VOWELS = "ou"      # Back rounded
    HARD_CONSONANTS = "fstkçşhp"  # Sert ünsüzler
    WHITESPACE = " \n\t"

    def __init__(self, reverse_dict):
        self.reverse_dict = reverse_dict

    def _starts_with_vowel(self, word: str) -> bool:
        """Check if word starts with a vowel."""
        return word and word[0] in self.ALL_VOWELS

    def _ends_with_vowel(self, word: str) -> bool:
        """Check if word ends with a vowel."""
        return word and word[-1] in self.ALL_VOWELS

    def _ends_with_any(self, word: str, charset: str) -> bool:
       # recursively check until first vowel starts from the end       
       i = len(word) - 1
       while i >= 0:
           if word[i] in charset:
               return True
           if word[i] in self.ALL_VOWELS:
               return False
           i -= 1
       return False

    def _ends_with_ince(self, word: str) -> bool:
        """Check if word ends with front vowels (ince ünlü)."""
        if word in ("saat", "kilovatsaat", "ziraat", "itaat"):
            return True
        # check until first vowel recursively
        return self._ends_with_any(word, self.INCE_VOWELS)

    def _ends_with_sert_unsuz(self, word: str) -> bool:
        """Check if word ends with a hard consonant."""
        return word and word[-1] in self.HARD_CONSONANTS

    def _get_vowel_suffix_index(self, prev_token: str) -> int:
        """Get suffix index based on vowel harmony rules."""
        if self._ends_with_any(prev_token, self.AI_VOWELS):
            return 0
        elif self._ends_with_any(prev_token, self.EI_VOWELS):
            return 1
        elif self._ends_with_any(prev_token, self.OU_VOWELS):
            return 2
        return 3

    def _select_correct_suffix(self, i: int, ids: List[int], prev_token: str) -> str:
        """Select the correct suffix based on morphological rules."""
        suffixes = self.reverse_dict[ids[i]]
        token_id = ids[i]        
        # Handle different suffix types with cleaner logic
        if token_id < 20013:
            # Basic suffix selection based on vowel harmony
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]
            
        elif token_id < 20023:  # nın, nin, nun, nün
            return suffixes[self._get_vowel_suffix_index(prev_token)]
            
        elif token_id == 20023:  # la, le, yla, yle
            end_of_word = True
            if i < len(ids) - 1:
                next_token = self.reverse_dict[ids[i + 1]][0]
                if next_token not in self.WHITESPACE:
                    end_of_word = False
            return self._handle_la_le_suffix(prev_token, suffixes, end_of_word)
            
        elif token_id <= 20025:  # da, de, ta, te, dan, den, tan, ten
            return self._handle_da_de_suffix(prev_token, suffixes)
            
        elif 20025 < token_id < 20029:  # dı, di, du, dü, tı, ti, tu, tü, etc.
            return self._handle_di_du_suffix(prev_token, suffixes)
            
        elif token_id == 20029:  # lık, lik, luk, lük, etc.
            return self._handle_lik_suffix(i, ids, prev_token, suffixes)
            
        elif token_id == 20030:  # cık, cik, cuk, cük, etc.
            return self._handle_cik_suffix(i, ids, prev_token, suffixes)
            
        elif token_id == 20031:  # mak, mek, may, mey
            return self._handle_mak_suffix(i, ids, prev_token, suffixes)
            
        elif token_id == 20032:  # acak, ecek, etc.
            return self._handle_acak_suffix(i, ids, prev_token, suffixes)
            
        return suffixes[0]

    def _handle_la_le_suffix(self, prev_token: str, suffixes: List[str], end_of_word: bool) -> str:
        """Handle la/le/yla/yle suffix selection."""
        if self._ends_with_vowel(prev_token) and end_of_word:
            return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
        else:
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

    def _handle_da_de_suffix(self, prev_token: str, suffixes: List[str]) -> str:
        """Handle da/de/ta/te suffix selection."""
        if self._ends_with_sert_unsuz(prev_token):
            return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
        return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

    def _handle_di_du_suffix(self, prev_token: str, suffixes: List[str]) -> str:
        """Handle dı/di/du/dü suffix selection."""
        base_index = self._get_vowel_suffix_index(prev_token)
        return suffixes[base_index + 4] if self._ends_with_sert_unsuz(prev_token) else suffixes[base_index]

    def _handle_lik_suffix(self, i: int, ids: List[int], prev_token: str, suffixes: List[str]) -> str:
        """Handle lık/lik/luk/lük suffix selection."""
        if i >= len(ids) - 1:
            return suffixes[0]
        
        next_token = self.reverse_dict[ids[i + 1]][0]
        base_index = self._get_vowel_suffix_index(prev_token)
        return suffixes[base_index + 4] if self._starts_with_vowel(next_token) else suffixes[base_index]

    def _handle_cik_suffix(self, i: int, ids: List[int], prev_token: str, suffixes: List[str]) -> str:
        """Handle cık/cik/cuk/cük suffix selection."""
        if i >= len(ids) - 1:
            return suffixes[0]
        
        next_token = self.reverse_dict[ids[i + 1]][0]
        base_index = self._get_vowel_suffix_index(prev_token)
        
        if self._starts_with_vowel(next_token):
            offset = 12 if self._ends_with_sert_unsuz(prev_token) else 8
        else:
            offset = 4 if self._ends_with_sert_unsuz(prev_token) else 0
        
        return suffixes[base_index + offset]

    def _handle_mak_suffix(self, i: int, ids: List[int], prev_token: str, suffixes: List[str]) -> str:
        """Handle mak/mek/may/mey suffix selection."""
        if i >= len(ids) - 1:
            return suffixes[0]
        
        next_token = self.reverse_dict[ids[i + 1]][0]
        base_index = 1 if self._ends_with_ince(prev_token) else 0
        return suffixes[base_index + 2] if self._starts_with_vowel(next_token) else suffixes[base_index]

    def _handle_acak_suffix(self, i: int, ids: List[int], prev_token: str, suffixes: List[str]) -> str:
        """Handle acak/ecek/yacak/yecek suffix selection."""
        is_vowel_ending = self._ends_with_vowel(prev_token)
        is_ince = self._ends_with_ince(prev_token)

        is_vowel_starting = False
        if i < len(ids) - 1:
          next_token = self.reverse_dict[ids[i + 1]][0]
          is_vowel_starting = self._starts_with_vowel(next_token)
        
        if is_vowel_starting:
            if is_vowel_ending:
                return suffixes[7] if is_ince else suffixes[6]
            else:
                return suffixes[3] if is_ince else suffixes[2]
        else:
            if is_vowel_ending:
                return suffixes[5] if is_ince else suffixes[4]
            else:
                return suffixes[1] if is_ince else suffixes[0]

    def _select_correct_root(self, i: int, ids: List[int]) -> str:
        """Select the correct root form based on morphological context."""
        token_id = ids[i]
        
        if i >= len(ids) - 2:
            return self.reverse_dict[token_id][0]
        
        next_token = self.reverse_dict[ids[i + 1]][0]
        
        if 100 <= token_id < 2080:
            if self._starts_with_vowel(next_token):
                return self.reverse_dict[token_id][1]
            elif token_id <= 110 and ids[i + 1] == 20034:
                return self.reverse_dict[token_id][2]
            else:
                return self.reverse_dict[token_id][0]
                
        elif 2080 <= token_id < 2315:
            if ids[i + 1] == 20021:  # yor
                return self.reverse_dict[token_id][1]
            else:
                return self.reverse_dict[token_id][0]
        
        return self.reverse_dict[token_id][0]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs to text."""
        if not ids:
            return ""
        
        text_parts = []
        i = 0
        
        while i < len(ids):
            token_id = ids[i]
            # Handle special tokens
            if token_id == 0 and i < len(ids) - 1:  # uppercase
                next_token = self.reverse_dict[ids[i + 1]][0]
                text_parts.append(next_token.capitalize())
                i += 2
                continue
            elif token_id == 1:  # unknown
                text_parts.append("▁u▁")
            elif token_id in self.reverse_dict:
                tokens = self.reverse_dict[token_id]
                if len(tokens) > 1 and i > 0:
                    if token_id < 20000:  # root token
                        text_parts.append(self._select_correct_root(i, ids))
                    else:  # suffix token
                        j = -1
                        prev_token = text_parts[j]
                        # while prev_token is not a word, get the previous token
                        while not prev_token.isalpha() and j > -len(text_parts):
                            prev_token = text_parts[j]
                            j -= 1
                        text_parts.append(self._select_correct_suffix(i, ids, prev_token))
                else:
                    text_parts.append(tokens[0])
            else:
                text_parts.append("▁")
            
            i += 1
        
        return "".join(text_parts)
