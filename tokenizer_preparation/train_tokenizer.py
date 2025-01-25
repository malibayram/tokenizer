import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# Paths
text_path = '/Users/alibayram/Desktop/zanai/UD_Turkish-BOUN/olds/combined_text.txt'

print("Loading dictionaries...")
# Load dictionaries
edatlar_dict = json.load(open('semantic_tokenizer/edatlar_v01.json', 'r', encoding='utf-8'))
ekler_dict = json.load(open('semantic_tokenizer/ekler_v03.json', 'r', encoding='utf-8'))
kokler_dict = json.load(open('semantic_tokenizer/kokler_v03.json', 'r', encoding='utf-8'))

combined_dict = {**edatlar_dict, **ekler_dict, **kokler_dict}
print(f"Total vocabulary size from dictionaries: {len(combined_dict)}")

# Initialize tokenizer
print("\nInitializing tokenizer...")
unk_token = '<unk>'
tokenizer = Tokenizer(BPE(unk_token=unk_token))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.add_tokens(list(combined_dict.keys()))

# Special tokens and trainer setup
spl_tokens = ['<bos>', '<eos>', '<unk>', '<pad>', '<start_of_turn>', '<end_of_turn>']
trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=10000, limit_alphabet=1200, min_frequency=64)

# Load texts
print("\nLoading training texts...")
with open(text_path, 'r', encoding='utf-8') as f:
    texts = f.read().lower().split('\n')
print(f"Loaded {len(texts):,} lines of text")

# Training
print("\nTraining tokenizer...")
def batch_iterator(texts, batch_size=100000):
    for i in tqdm(range(0, len(texts), batch_size), desc="Training progress"):
        yield texts[i:i + batch_size]

tokenizer.train_from_iterator(batch_iterator(texts), trainer=trainer)

# Save tokenizer
print("\nSaving tokenizer...")
tokenizer.save("custom_bpe_tokenizer3.json")

vocab_size = tokenizer.get_vocab_size()
print(f"\nFinal vocabulary size: {vocab_size:,}")
print("Training completed successfully!") 

""" 
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# Paths
text_path = '/Users/alibayram/Desktop/zanai/UD_Turkish-BOUN/olds/combined_text.txt'

print("Loading dictionaries...")
# Load dictionaries
edatlar_dict = json.load(open('semantic_tokenizer/edatlar_v01.json', 'r', encoding='utf-8'))
ekler_dict = json.load(open('semantic_tokenizer/ekler_v03.json', 'r', encoding='utf-8'))
kokler_dict = json.load(open('semantic_tokenizer/kokler_v03.json', 'r', encoding='utf-8'))

combined_dict = {**edatlar_dict, **ekler_dict, **kokler_dict}
print(f"Total vocabulary size from dictionaries: {len(combined_dict)}")

# Initialize tokenizer
print("\nInitializing tokenizer...")
unk_token = '<unk>'
tokenizer = Tokenizer(BPE(unk_token=unk_token))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.add_tokens(list(combined_dict.keys()))

# Special tokens and trainer setup
spl_tokens = ['<bos>', '<eos>', '<unk>', '<pad>', '<start_of_turn>', '<end_of_turn>']
trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=10000, limit_alphabet=1200, min_frequency=64)

tokenizer.train(files=[text_path], trainer=trainer)

# Save tokenizer
print("\nSaving tokenizer...")
tokenizer.save("custom_bpe_tokenizer2.json")

vocab_size = tokenizer.get_vocab_size()
print(f"\nFinal vocabulary size: {vocab_size:,}")
print("Training completed successfully!") 
 """