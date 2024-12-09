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

### Grammatical Event Revertor 

There are nine grammatical events in Turkish language:
    
    - Ünlü Daralması (Vowel Contraction)
    - Ünlü Düşmesi (Vowel Fall)
    - Ünsüz Düşmesi (Consonant Fall)
    - Ünsüz Türemesi (Consonant Derivation)
    - Ünsüz Yumuşaması (Consonant Softening)
    - Ünsüz Sertleşmesi (Consonant Hardening)
    - Kaynaştırma (Suffix Welding)

We have to revert any word that has passed a grammatical event and obtain root and suffixes.

Examples:

    Geldiği-> Gel (root) + di (past tense suffix)  + ğ (suffix welding) + i (to point suffix)


## **To be continued**
    

