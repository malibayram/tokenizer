import argparse
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple

from tr_decoder import TRDecoder
from tr_gpt_decoder import TRGPTTokenizer


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

        self.decoder = TRDecoder(self.reverse_dict)
        self.gpt_decoder = TRGPTTokenizer(self.reverse_dict)

        self.roots = roots
        self.suffixes = suffixes
        self.bpe_tokens = bpe_tokens
        self.max_root_len = max(len(k) for k in roots) if roots else 0
        self.max_suffix_len = max(len(k) for k in suffixes) if suffixes else 0
        self.max_bpe_len = max(len(k) for k in bpe_tokens) if bpe_tokens else 0
        
        self.uppercase_marker = {"token": "<uppercase>", "id": 0, "type": TokenType.ROOT}
        self.unknown_marker = {"token": "<unknown>", "id": 1, "type": TokenType.ROOT}
        self.space_marker = {"token": " ", "id": 2, "type": TokenType.ROOT}

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

    def decode(self, ids: List[int]) -> str:
        return TRDecoder(self.reverse_dict).decode(ids)
    
    def gpt_decode(self, ids: List[int]) -> str:
        return self.gpt_decoder.decode(ids)
    

def main(text: str = ""):
    tokenizer = TRTokenizer()
    if text == "":
        text = """
        Bugün Ali, sabah erken saatte İstanbul’un sakin bir köyünün kıyısındaki eski evinin kapısını yavaşça açıp bahçeye çıktı. Kitaptan not aldığı fikrini deftere yazdı ve “Bu projenin aşamaları uç uca eklenecek, sonra rapor olarak tek tek sunulacak,” dedi. Öğleye doğru kulüpte kısa bir toplantı yaptı; şefle ve arkadaşıyla yüz yüze görüşüp gelişecek işleri ayrıntılıca konuştu.
        Toplantıdan sonra taslağı evde tamamlayacak işleri listeleyecek ve yapacağı deneyleri planlayacaktı; ancak ağaçtan düşen küçücük bir dalcık ayağına değince bahçede kalıp ölçümleri bitirmeyi seçti. Yorgunluğu artınca, “Bu metnin ilk bölümünü kısacık tutayım, sonra uzunca açıklayayım,” diye düşündü.
        Akşamüstü arabacı komşusunun getirdiği parçalık malzemeyi atölyede denedi; taşçı ustanın önerisiyle küçücük delikleri büyütüp ince ayarlı bir ölçümcük daha ekledi. Nihayet geliyor gibi görünen sonuçlar onu sevindirdi; “Bu verilerin çoğunu yarın yeniden deneyip doğrulayacağım; yapmayıp beklersem geçecek zamanı boşa harcamış olurum,” dedi.
        """
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
    print(tokenizer.gpt_decode(ids))

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, default="")
    args = parser.parse_args()
    main(args.text)