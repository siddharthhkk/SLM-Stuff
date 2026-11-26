import regex as re
from Tokenizer import get_states, merge, Tokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_patterns = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_patterns, text)
        tokens = [list(ch.encode('utf-8')) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in tokens:
                get_states(chunk_ids, stats)     # passing in stats will update it in place, adding up counts
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            tokens = [merge(chunk_ids, top_pair, idx) for chunk_ids in tokens]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurences")
        self.merges = merges
        self.vocab = vocab

    def register_special_token(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, tokens):
        part_bytes = []
        for idx in tokens:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f'invalid token id: {idx}')
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode('utf-8', errors='replace')

    def _encode_chunks_(self, text_bytes):  #return the tokens ids
        tokens = list(text_bytes)
        while len(tokens) >= 2:
            state = get_states(tokens)
            top_pair = min(state, key=lambda p: self.merges.get(p, float('inf')))
            if top_pair not in self.merges:
                break
            idx = self.merges[top_pair]
            tokens = merge(tokens, top_pair, idx)
        return tokens

    def _encode_ordinary(self, text):    #encoding that ignores special characters
        text_chunks = re.findall(self.compiled_patterns, text)
        tokens = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self._encode_chunks_(chunk_bytes)
            tokens.extend(chunk_ids)
        return tokens

    def encode_ordinary(self, text):
        return self._encode_ordinary(text)

    def encode(self, text, allowed_special='none_raise'):
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:    # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text, we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens, we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text all chunks of text are encoded separately, then results are joined
        tokens = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                tokens.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                tokens.extend(self.encode_ordinary(part))
        return tokens