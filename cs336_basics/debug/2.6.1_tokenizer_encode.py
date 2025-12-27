from cs336_basics.debug.tokenizer import Tokenizer
from pathlib import Path

BPE_DIR = Path("/home/dexterding/projects/assignment1-basics/bpe_outputs")

OWT_MERGE_FILE = BPE_DIR /"owt_merges.pkl"
OWT_VOCAB_FILE = BPE_DIR / "owt_vocab.pkl"

# print(OWT_MERGE_FILE)
# print(OWT_VOCAB_FILE)
test_string = "the cat ate"
# tokenizer = Tokenizer.from_files(vocab_filepath=OWT_VOCAB_FILE,merges_filepath=OWT_MERGE_FILE)


vocab ={0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at'}

merges = [
         (b't',b'h'),
         (b' ',b'c'),
         (b' ',b'a'),
         (b'th',b'e'),
         (b' a',b't')]
special_tokens = ["<|speical|>"] 
tokenizer = Tokenizer(vocab=vocab,merges=merges)

# pre_tokenized_list = tokenizer._pre_tokenize(test_string)
# print(pre_tokenized_list)
encoding_string = tokenizer.encode(test_string)
print(encoding_string)
