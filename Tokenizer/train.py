import os
import time
from Tokenizer import Tokenizer
from regexTokenizer import RegexTokenizer

text = open('taylorswift.txt', 'r', encoding='utf-8').read()

os.makedirs("models", exist_ok=True)

t0 = time.time()

for TokenizerClass, name in zip([Tokenizer, RegexTokenizer], ['Basic', "regex"]):
    tokenizerr = TokenizerClass()
    tokenizerr.train(text, 512, verbose=True)
    prefix = os.path.join('models', name)
    tokenizerr.save(prefix)

t1 = time.time()
print(f'Training took {t1 - t0:.2f}')