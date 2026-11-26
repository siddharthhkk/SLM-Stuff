import unicodedata

def get_states(tokens: list, counts=None)-> dict:
    counts = {} if counts is None else counts
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(tokens: list, pair: tuple, idx: int)-> list:
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def replace_cntrl_chars(s: str)-> str:       # control character arelike \n, \t, \r, or null \x00
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":   # not a control character!
            chars.append(ch)  
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes)-> str:
    s = t.decode('utf-8', errors='replace')
    return replace_cntrl_chars(s)     # prints the string without control tokens


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_states(tokens)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = merge(tokens, top_pair, idx)
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurences")
        self.merges = merges
        self.vocab = vocab

    def decode(self, tokens):
        text_bytes = b"".join(self.vocab[idx] for idx in tokens)
        return text_bytes.decode('utf-8', errors='replace')

    def encode(self, text):
        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)
        while len(tokens) >= 2:
            stats = get_states(tokens)
            # subtle: if there are no more merges available, the key will result in an inf for every single pair, and the min will be
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))  
            if pair not in self.merges:
                break
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]  
            tokens = merge(tokens, pair, idx)
        return tokens
                

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        return vocab


    def save(self, file_prefix):
        model_file = file_prefix + ".model"       # write the model: to be used in load() later
        with open(model_file, 'w') as f:
            f.write('minbpe v1\n')                # write the version, pattern and merges
            f.write(f'{self.pattern}\n')
            # write the special tokens, first the number of them, then each one
            f.write(f'{len(self.special_tokens)}\n')  
            for special, idx in self.special_tokens.items():
                f.write(f'{special} {idx}\n')
            for idx1, idx2 in self.merges:
                f.write(f'{idx1} {idx2}\n')

        # write the vocab fole
        vocab_file = file_prefix + '.vocab'
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences and cannot be decoded into valid strings. 
                # Here we're using errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load() because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith('.model')
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding='utf-8') as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().rsplit(maxsplit=1)
                special_tokens[special] = int(special_idx)

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()