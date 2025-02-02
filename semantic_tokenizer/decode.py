import json
import os

class SemanticDecoder:
    def __init__(self):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load necessary resources
        with open(os.path.join(current_dir, "kokler_v02.json"), "r", encoding='utf-8') as f:
            self.roots = json.load(f)
            f.close()

        with open(os.path.join(current_dir, "ekler_v02.json"), "r", encoding='utf-8') as f:
            self.suffixes = json.load(f)
            f.close()

        self.semantic_tokens = {
            "__PLU__": ["ler", "lar"],  
            "__ABL__": ["den", "dan", "ten", "tan"],  
            "__LOC__": ["de", "da", "te", "ta"],  
            
        }

        self.unknown = "<UNKNOWN>"
        self.space = "<SPACE>"

        # Vowel sets for harmony rules
        self.front_vowels = set('eiöü')
        self.back_vowels = set('aıou')
        self.rounded_vowels = set('oöuü')
        self.unrounded_vowels = set('aeıi')
        self.all_vowels = self.front_vowels.union(self.back_vowels)

        # Consonant sets for hardening rules
        self.hardening_consonants = set('pçtksşhf')
        
    # Finds the last vowel in the word/ Kelimenin son sesli harfini bulur
    def _get_last_vowel(self, word):
        """Get the last vowel in the word"""
        for char in reversed(word):
            if char in self.all_vowels:
                return char
        return None

    def _get_last_char(self, word):
        """Get the last character of the word"""
        return word[-1] if word else ''

    def _needs_hard_suffix(self, word):
        """Check if the word needs a hard consonant suffix"""
        last_char = self._get_last_char(word)
        return last_char in self.hardening_consonants

    def _get_turkish_suffix(self, semantic_token, root):
        """
        Convert semantic token to appropriate Turkish suffix based on vowel harmony
        and consonant hardening rules
        """
        if semantic_token not in self.semantic_tokens:
            return semantic_token

        possible_suffixes = self.semantic_tokens[semantic_token]
        last_vowel = self._get_last_vowel(root)

        if not last_vowel:
            return possible_suffixes[0]

        # First apply vowel harmony rule
        if last_vowel in self.front_vowels:
            # (e/i)
            base_suffix = next(suffix for suffix in possible_suffixes if 'e' in suffix)
        else:
            #(a/ı)
            base_suffix = next(suffix for suffix in possible_suffixes if 'a' in suffix)

        # Then apply consonant hardening if needed
        if self._needs_hard_suffix(root):
            # Convert 'd' to 't' at the start of suffix
            if base_suffix.startswith('d'):
                base_suffix = 't' + base_suffix[1:]

        return base_suffix

    def _handle_suffix_welding(self, root, suffix):
        """
        Handle suffix welding cases (kaynaştırma)
        """
        # Get the last character of the root and first character of the suffix
        root_end = root[-1] if root else ''
        suffix_start = suffix[0] if suffix else ''
        
        # If root ends with a vowel and suffix starts with a vowel
        if root_end in self.all_vowels and suffix_start in self.all_vowels:
            return root + 'y' + suffix
        
        # If root ends with 'k' and suffix starts with a vowel
        if root_end == 'k' and suffix_start in self.all_vowels:
            return root[:-1] + 'ğ' + suffix
        # for default case just concatenate root and suffix
        return root + suffix

    def decode(self, tokens):
        """
        Decode a list of semantic tokens back into Turkish word
        
        Args:
            tokens: List of tokens (root + semantic tokens)
            
        Returns:
            str: Reconstructed Turkish word
        """
        if not tokens:
            return ""

        # First token is always the root
        result = tokens[0]

        # Process remaining semantic tokens
        for token in tokens[1:]:
            if token == self.unknown:
                continue
                
            # Convert semantic token to appropriate Turkish suffix
            turkish_suffix = self._get_turkish_suffix(token, result)
            
            # Apply suffix welding if needed
            result = self._handle_suffix_welding(result, turkish_suffix)

        return result

# Example usage
if __name__ == "__main__":
    decoder = SemanticDecoder()
    
    # Examples
    test_cases = [
        (["ev", "__PLU__", "__LOC__"], "evlerde"), 
        (["kitap", "__PLU__", "__ABL__"], "kitaplardan"),  
        (["ağaç", "__PLU__"], "ağaçlar"),  
        (["demir", "__LOC__"], "demirde"),  
        (["araba", "__LOC__"], "arabada"),  
        (["araba", "__ABL__"], "arabadan"),  
        (["araba", "__PLU__", "__ABL__"], "arabalardan"),  
        (["araba", "__PLU__", "__LOC__"], "arabalarda"),  

    ]
    
    for tokens, expected in test_cases:
        result = decoder.decode(tokens)
        print(f"Input tokens: {tokens}")
        print(f"Decoded word: {result}")
        print(f"Expected: {expected}")
        print("---") 
        