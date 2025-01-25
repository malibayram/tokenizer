def yumusama_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    yumusak_kelimeler = []

    for line in lines:
        words = line.strip().split()
        for word in words:
            yumusak_kelimeler.append(replace_char(word))

    return yumusak_kelimeler

def load_exceptions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        exceptions = set(line.strip() for line in f.readlines())
    return exceptions

def replace_char(word, exceptions):
    replacement_map = {'ç': 'c', 'k': 'ğ', 't': 'd', 'p': 'b'}

    if word in exceptions:
        return word

    if word and word[-1] in replacement_map:
        if len(word) > 1:
            if word[-1] == 'k' and word[-2] == 'n':
                replacement_map = {'ç': 'c', 'k': 'g', 't': 'd', 'p': 'b'}
        return word[:-1] + replacement_map[word[-1]]
    return word

def yumusama_write(file_path, exceptions_file_path):

    exceptions = load_exceptions(exceptions_file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        words = line.strip().split()
        for word in words:
            updated_lines.append(word)
            modified_word = replace_char(word, exceptions)
            if modified_word != word:
                updated_lines.append(modified_word)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in updated_lines:
            f.write(item + '\n')


def decode_yumusat(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in lines:
            f.write(item + ' ')

file_path = "veri/deneme.txt"
exceptions_file_path = "veri/yumusama_istisna.txt"
yumusama_write(file_path, exceptions_file_path)
decode_yumusat(file_path)
