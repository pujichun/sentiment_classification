from zhon.hanzi import punctuation

def remove_punctuation(text:str) -> str:
    s = ''
    for c in text:
        if c not in punctuation:
            s += c
    return s
