from typing import List, Optional, Tuple


class TRGPTTokenizer:
    # --- Phonology sets ---
    VOWELS_ALL = set("aeıioöuü")
    FRONT = set("eiöü")     # ince ünlüler
    AI = set("aı")
    EI = set("ei")
    OU = set("ou")
    SERT_UNSUZ = set("fstkçşhp")

    # --- Special token ids (avoid magic numbers) ---
    T_UPPER = 0
    T_SPACE = 1
    T_NEWLINE = 2
    T_TAB = 3
    T_UNKNOWN = 4

    # Suffix / root bands (keep your semantics, just named)
    BAND_FRONT_BACK = (0, 20013)         # ids < 20013 → 2-way (front/back) selection
    BAND_NIN_SET    = (20013, 20023)     # nın/nin/nun/nün set
    T_LA_LE_YLA     = 20023              # la/le/yla/yle
    BAND_DA_TE      = (20024, 20025)     # da,de,ta,te + dan,den,tan,ten (<=20025)
    BAND_DI_TI      = (20026, 20028)     # (20025, 20029) exclusive upper bound in your code
    T_LIK           = 20029              # lık/lik/luk/lük + ğ’lı varyantlar
    T_CIK           = 20030              # cık/cik/cuk/cük/çık/… + ğ’lı varyantlar
    T_MAK_MEK       = 20031              # mak/mek/may/mey
    T_ACAK_ECEK     = 20032              # acak/ecek/…
    T_CIK_SET_ID    = 20034              # your comment references this in _select_correct_root

    # Root categories (from your docstring)
    ROOT_MIN = 100
    ROOT_UNLU_DUSME_MAX = 110
    ROOT_KALIN_GENISLEME_MIN = 2080
    ROOT_INCORRECT_MAX = 2315

    def __init__(self, reverse_dict):
        # reverse_dict: id -> [form0, form1, ...]
        self.reverse_dict = reverse_dict

    # ---------- Small safe helpers ----------
    @staticmethod
    def _ends_with_any(word: str, charset: set, k: int = 3) -> bool:
        return bool(word) and any(c in charset for c in word[-k:])

    @staticmethod
    def _first(s: List[str], default: str = "") -> str:
        return s[0] if s else default

    def _tok_str(self, tid: int) -> str:
        return self._first(self.reverse_dict.get(tid, [""]))

    def _prev_token(self, ids: List[int], i: int, skip_ws: bool = True) -> Tuple[str, Optional[int]]:
        j = i - 1
        if skip_ws:
            while j >= 0 and ids[j] in (self.T_SPACE, self.T_NEWLINE, self.T_TAB):
                j -= 1
        if j >= 0:
            return self._tok_str(ids[j]), j
        return "", None

    def _next_token(self, ids: List[int], i: int, skip_ws: bool = True) -> Tuple[str, Optional[int]]:
        j = i + 1
        if skip_ws:
            while j < len(ids) and ids[j] in (self.T_SPACE, self.T_NEWLINE, self.T_TAB):
                j += 1
        if j < len(ids):
            return self._tok_str(ids[j]), j
        return "", None

    # ---------- Phonology checks (safe) ----------
    def _starts_with_vowel(self, word: str) -> bool:
        return bool(word) and word[0].lower() in self.VOWELS_ALL

    def _ends_with_vowel(self, word: str) -> bool:
        return bool(word) and word[-1].lower() in self.VOWELS_ALL

    def _ends_with_ince(self, word: str) -> bool:
        return self._ends_with_any(word.lower(), self.FRONT)

    def _ends_with_ai(self, word: str) -> bool:
        return self._ends_with_any(word.lower(), self.AI)

    def _ends_with_ei(self, word: str) -> bool:
        return self._ends_with_any(word.lower(), self.EI)

    def _ends_with_ou(self, word: str) -> bool:
        return self._ends_with_any(word.lower(), self.OU)

    def _ends_with_sert_unsuz(self, word: str) -> bool:
        return bool(word) and word[-1].lower() in self.SERT_UNSUZ

    # ---------- Selection logic ----------
    def _select_correct_suffix(self, i: int, ids: List[int]) -> str:
        sid = ids[i]
        suffixes = self.reverse_dict[sid]  # assume exists
        prev_token, _ = self._prev_token(ids, i, skip_ws=True)

        # ids < 20013 and i > 0  →  front/back (2 variants)
        if sid < self.BAND_FRONT_BACK[1] and i > 0:
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

        # 20013 ≤ id < 20023  → nın, nin, nun, nün (4 variants)
        if self.BAND_NIN_SET[0] <= sid < self.BAND_NIN_SET[1] and i > 0:
            if   self._ends_with_ai(prev_token): return suffixes[0]
            elif self._ends_with_ei(prev_token): return suffixes[1]
            elif self._ends_with_ou(prev_token): return suffixes[2]
            else:                                return suffixes[3]

        # id == 20023 → la, le, yla, yle (need vowel-start and front/back)
        if sid == self.T_LA_LE_YLA and i > 0:
            if self._ends_with_vowel(prev_token):
                return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

        # id ≤ 20025 → da, de, ta, te, dan, den, tan, ten (4 variants)
        if sid <= self.BAND_DA_TE[1] and i > 0:
            if self._ends_with_sert_unsuz(prev_token):
                return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

        # 20025 < id < 20029 → voiced/unvoiced + (aı/ei/ou/öü) (8 variants)
        if self.BAND_DI_TI[0] < sid < 20029 and i > 0:
            if self._ends_with_sert_unsuz(prev_token):
                if   self._ends_with_ai(prev_token): return suffixes[4]
                elif self._ends_with_ei(prev_token):  return suffixes[5]
                elif self._ends_with_ou(prev_token):  return suffixes[6]
                else:                                 return suffixes[7]
            else:
                if   self._ends_with_ai(prev_token): return suffixes[0]
                elif self._ends_with_ei(prev_token):  return suffixes[1]
                elif self._ends_with_ou(prev_token):  return suffixes[2]
                else:                                 return suffixes[3]

        # id == 20029 → lık/lik/luk/lük + ğ’lı varyantlar (8 variants)
        if sid == self.T_LIK and i > 0:
            next_token, _ = self._next_token(ids, i, skip_ws=True)
            vowel_next = self._starts_with_vowel(next_token)
            if vowel_next:
                if   self._ends_with_ai(prev_token): return suffixes[4]
                elif self._ends_with_ei(prev_token):  return suffixes[5]
                elif self._ends_with_ou(prev_token):  return suffixes[6]
                else:                                 return suffixes[7]
            else:
                if   self._ends_with_ai(prev_token): return suffixes[0]
                elif self._ends_with_ei(prev_token):  return suffixes[1]
                elif self._ends_with_ou(prev_token):  return suffixes[2]
                else:                                 return suffixes[3]

        # id == 20030 → cık/cik/… + ğ’lı varyantlar (16 variants)
        if sid == self.T_CIK and i > 0:
            next_token, _ = self._next_token(ids, i, skip_ws=True)
            starts_vowel = self._starts_with_vowel(next_token)
            hard = self._ends_with_sert_unsuz(prev_token)
            # offset base depending on (starts_vowel, hard)
            # mapping to your indices:
            #   not vowel & not hard: [0..3]
            #   not vowel & hard    : [4..7]
            #   vowel & not hard    : [8..11]
            #   vowel & hard        : [12..15]
            base = (8 if starts_vowel else 0) + (4 if hard else 0)
            if   self._ends_with_ai(prev_token): return suffixes[base + 0]
            elif self._ends_with_ei(prev_token):  return suffixes[base + 1]
            elif self._ends_with_ou(prev_token):  return suffixes[base + 2]
            else:                                 return suffixes[base + 3]

        # id == 20031 → mak/mek/may/mey (4 variants)
        if sid == self.T_MAK_MEK and i > 0:
            next_token, _ = self._next_token(ids, i, skip_ws=True)
            if self._starts_with_vowel(next_token):
                return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
            return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

        # id == 20032 → acak/ecek + y-’li varyantlar (8 variants)
        if sid == self.T_ACAK_ECEK and i > 0:
            next_token, _ = self._next_token(ids, i, skip_ws=True)
            prev_vowel = self._ends_with_vowel(prev_token)
            next_vowel = self._starts_with_vowel(next_token)
            if next_vowel:
                if prev_vowel:
                    return suffixes[7] if self._ends_with_ince(prev_token) else suffixes[6]
                else:
                    return suffixes[3] if self._ends_with_ince(prev_token) else suffixes[2]
            else:
                if prev_vowel:
                    return suffixes[5] if self._ends_with_ince(prev_token) else suffixes[4]
                else:
                    return suffixes[1] if self._ends_with_ince(prev_token) else suffixes[0]

        # Fallback
        return suffixes[0]

    def _select_correct_root(self, i: int, ids: List[int]) -> str:
        rid = ids[i]
        forms = self.reverse_dict[rid]

        # 100 ≤ id < 2080
        if self.ROOT_MIN <= rid < self.ROOT_KALIN_GENISLEME_MIN and i < len(ids) - 1:
            next_token, _ = self._next_token(ids, i, skip_ws=True)
            if self._starts_with_vowel(next_token):
                return forms[1] if len(forms) > 1 else forms[0]
            # özel durum: (≤110) + cık seti
            if rid <= self.ROOT_UNLU_DUSME_MAX and i + 1 < len(ids) and ids[i + 1] == self.T_CIK_SET_ID:
                return forms[2] if len(forms) > 2 else forms[0]
            return forms[0]

        # 2080 ≤ id < 2315
        if self.ROOT_KALIN_GENISLEME_MIN <= rid < self.ROOT_INCORRECT_MAX and i < len(ids) - 1:
            # özel durum: +yor
            if i + 1 < len(ids) and ids[i + 1] == 20021:  # yor
                return forms[1] if len(forms) > 1 else forms[0]
            return forms[0]

        # default
        return forms[0]

    # ---------- Decoder ----------
    def decode(self, ids: List[int]) -> str:
        text = []
        i = 0
        n = len(ids)
        while i < n:
            tid = ids[i]

            if tid == self.T_UPPER and i + 1 < n:
                token = self._tok_str(ids[i + 1])
                text.append(token.capitalize())
                i += 2
                continue

            if tid == self.T_UNKNOWN:
                text.append("▁u▁"); i += 1; continue

            if tid in self.reverse_dict:
                forms = self.reverse_dict[tid]
                if len(forms) > 1 and i > 0:
                    if tid < 20000:  # root
                        text.append(self._select_correct_root(i, ids))
                    else:           # suffix
                        text.append(self._select_correct_suffix(i, ids))
                else:
                    text.append(forms[0])
            else:
                text.append("▁")
            i += 1

        return "".join(text)