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
        token_ids.append("<upper>")
        word = word.lower()
      if "ler" or "lar" in word:
        subwords = word.split("ler")
        subwords = word.split("lar")
        token_ids.append(self.tokenizer.encode(subwords[0]))
        token_ids.append("<plural>")
        token_ids.append(self.tokenizer.encode(subwords[1]))
      else:
        token_ids.append(self.tokenizer.encode(word))
    return token_ids
  
  def decode(self, token_ids: list):
    tokens = []
    for token_id in token_ids:
      tokens.append(self.tokenizer.decode(token_id))
      for i, token in enumerate(tokens):
        if token == "<upper>":
          tokens[i+1] = tokens[i+1].capitalize()
          tokens.pop(i)
        if token == "<plural>":
          tokens[i-1] += "ler"
          tokens.pop(i)
    output = " ".join(tokens)
    return output