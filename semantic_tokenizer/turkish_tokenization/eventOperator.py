# This python file contains the EventOperator class which is used to create event operators for the event extraction process.

# Useful arrays   

hard_consonants = ['f', 's', 't', 'k', 'ç', 'ş', 'h', 'p']
vowels = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']

# The EventOperator class is used to create event operators for the event process.
class EventOperator:
    
    # Consonant Hardening (Ünsüz Sertleşmesi) Function to harden the consonants in the given word.
    def sertlestirme_encode(self, root, suffix):
        toBeHardened = {'c': 'ç', 'd': 't', 'g': 'k'}
        if root[-1] in consonants and suffix[0] in toBeHardened:
            word = root + toBeHardened[suffix[0]] + suffix[1:]
        return word
    
    # Revert Consonant Hardening (Ünsüz Sertleşmesi Geri Dönüş) Function to revert the consonant hardening in the given word.
    def sertlestirme_decode(self, word):
        toBeReverted = {'ç': 'c', 't': 'd', 'k': 'g'}
