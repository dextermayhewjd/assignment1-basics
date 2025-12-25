import pickle
import regex as re 


class Tokenizer:
    PAT2 = re.compile(r"""
        '(?:[sdmt]|ll|ve|re)
        | \ ?\p{L}+
        | \ ?\p{N}+
        | \ ?[^\s\p{L}\p{N}]+
        | \s+(?!\S)
        | \s+
        """, re.VERBOSE)
    
    def __init__(self,
                 vocab:dict[int,bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        
    @classmethod
    def from_files(
                   cls, 
                   vocab_filepath: str,
                   merges_filepath:str,
                   special_tokens: list[str] |None=None 
                   ):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # def encode(self, text: str)-> list[int]:

    # def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
    #   s 
    # def decode(self, ids:list[int])-> str:
    #   s 