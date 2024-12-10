# Semantic Tokenizer Project

#### *This project will progress in both English and Turkish for global reach purposes.*

## Documentation for Turkish Work

The semantic tokenization for Turkish have two main parts:

    1. Encode
    2. Decode 

### Encode Steps

*Word -> Root + Suffix -> Root + Semantic Token*

We have two main process in this flow:

    1. Grammatical Event Handler - Encode (first arrow)
    2. Semantic Converter (second arrow)

#### Grammatical Event Handler - Encode (GEH-E)

There are nine grammatical events in Turkish language:
    
    - Ünlü Daralması (Vowel Contraction)
    - Ünlü Düşmesi (Vowel Fall)
    - Ünsüz Düşmesi (Consonant Fall)
    - Ünsüz Türemesi (Consonant Derivation)
    - Ünsüz Yumuşaması (Consonant Softening)
    - Ünsüz Sertleşmesi (Consonant Hardening)
    - Kaynaştırma (Suffix Welding)

Grammatical Event Handler reverts any word that has passed a grammatical event and obtain root and suffixes.

Example:

    Geldiği -> Gel (root) + di (past tense suffix)  + ğ (suffix welding) + i (to point suffix)


#### Semantic Converter 

The semantic converter will convert suffixes that GEH extracted from initial word to language-independent special tokens that carries the meanings of those subtokens.

Example: 

    Gel (root) + di (past tense suffix) + ğ (suffix welder) + i (pointing suffix) -> Gel + <past-sufx> + <point-sufx>

*Do not forget that we don't convert suffix welder part to a special token because that has no meaning so in encodig functions we just ignore those. It's going to be explained how to correctly decode words with suffix welding at the Decode Steps part.*

### Decode Steps

*Root + Semantic Token -> Root + Suffix -> Word*

We are going to use reverse version of two components that we used at the previous part:

    1. Semantic Converter (first arrow)
    2. Grammatical Event Handler - Decode (second arrow)

#### Semantic Converter 

The semantic converter will convert semantic tokens to related suffixes of Turkish language. 

Example:

    - Gel + <past-sufx> + <point-sufx> -> Gel (root) + di (past tense suffix) + i (pointing suffix)
    - Al + <past-sufx> + <plural-sufx> -> Al (root) + di (past tense suffix) + ler (plurality suffix)


*The semantic converter does not process any grammatical event such as "vowel compability"  and because of that we do have **Al + di + ler** insted of **Al + dı + lar** at this step. Rules like vowel compability will be processed at  Grammatical Event Handler (GEH) step.*

#### Grammatical Event Handler - Decode (GEH-D)

Grammatical Event Handler builds the final word from given input by merging root and suffixes in according to grammatical rules of preferred output language.

Example: 

    - Gel (root) + di (past tense suffix) + i (pointing suffix) -> Geldiği 
    - Al (root) + di (past tense suffix) + ler (plurality suffix) -> Aldılar 
 
*You can see that we have our desired form of output word after this final step. Further explanations of all components like GEH-E/D or Semantic Converter will be explained at their own folder in this repository.*