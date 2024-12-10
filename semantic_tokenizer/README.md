# Semantic Tokenizer Project

#### *This project will progress in both English and Turkish for global reach purposes.*

## Documentation for Turkish Work

The semantic tokenization for Turkish have two main parts:

    1. Encode
    2. Decode 

### Encode Steps

*Word -> Root + Suffix -> Root + Semantic Token*

We have two main process in this flow:

    1. Grammatical Event Revertor (firs arrow)
    2. Semantic Converter (second arrow)

### Grammatical Event Revertor (GER)

There are nine grammatical events in Turkish language:
    
    - Ünlü Daralması (Vowel Contraction)
    - Ünlü Düşmesi (Vowel Fall)
    - Ünsüz Düşmesi (Consonant Fall)
    - Ünsüz Türemesi (Consonant Derivation)
    - Ünsüz Yumuşaması (Consonant Softening)
    - Ünsüz Sertleşmesi (Consonant Hardening)
    - Kaynaştırma (Suffix Welding)

We have to revert any word that has passed a grammatical event and obtain root and suffixes.

Example:

    Geldiği -> Gel (root) + di (past tense suffix)  + ğ (suffix welding) + i (to point suffix)


### Semantic Converter 

The semantic converter will convert suffixes that GER extracted from initial word to language-independent special tokens that carries the meanings of those subtokens.

Example: 

    Gel (root) + di (past tense suffix) + ğ (suffix welder) + i (to point suffix) -> Gel + <past-sufx> + <point-sufx>

*Do not forget that we don't convert suffix welder part to a special token because that has no meaning so in encodig functions we just ignore those. It's going to be explained how to correctly decode words with suffix welding at the Decode Steps part.*

**To Be Continued**

