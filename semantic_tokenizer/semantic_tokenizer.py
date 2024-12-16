from transformers import AutoTokenizer

class SemanticTokenizer:
  def __init__(self, tokenizer: AutoTokenizer):
    self.tokenizer = tokenizer

  def tokenize(self, text: str):
    return self.tokenizer.tokenize(text)
  
  def encode(self, text: str):
    words = text.split()
    token_ids = []
    for word in words:
      # check if capitalized
      if word[0].isupper():
        token_ids.append(0)
        word = word.lower()
      
      token_ids += self.tokenizer.encode(word)
    return self.tokenizer.encode(text)
  
  def decode(self, token_ids: list):
    return self.tokenizer.decode(token_ids)