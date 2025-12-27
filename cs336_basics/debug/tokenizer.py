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
        self.bytes2int:dict[bytes,int] = {v:k for k,v in self.vocab.items()}
        
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

    def _encode_one_token(self,token:list[bytes])->list[int]:
        print("INPUT:", token)

        tokens = list(token) # 防御式编程

        
        for a,b in self.merges:               
            i = 0
            new_tokens = []
            
            while i < len(tokens):
                if i+1 <len(tokens) and tokens[i]==a and tokens[i+1] ==b:
                    new_tokens.append(a+b)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
                    
            tokens = new_tokens
        # print(tokens)    
        return [self.bytes2int[t] for t in tokens]
        
    
    def encode(self, text: str)-> list[int]:
        ids = []
        for bytes_seq in self._pre_tokenize(text=text):
            ids.extend(self._encode_one_token(bytes_seq))
        return ids

    # def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
    #   s 
    # def decode(self, ids:list[int])-> str:
    #   s     
    # 
    
    #"You are cat<|speical|>" -> ["You are cat","<|speical|>"]
    def _split_on_special_tokens(self,
                                 text: str
                                 ) -> list[str]:
        if not self.special_tokens:
            return [text]
        # pattern = "|".join(re.escape(tok) for tok in special_tokens)
        #关键：加 () 捕获组，让 special token 也出现在 split 结果里
        pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        
        return [p for p in re.split(pattern, text) if p]
    
    # ["You are cat","<|speical|>"] 
    # -> 
    # [[b't', b'h', b'e'], 
    #   [b' ', b'c', b'a', b't'], 
    #   [b' ', b'a', b't', b'e'], 
    #   b'<|special|>']
    def _pre_tokenize(self,
                      text:str, 
                      )->list[list[bytes]]:
            
        pre_tokenized_list = []
        
        chunks = self._split_on_special_tokens(text)
        # print(chunks)
        special_set = set(self.special_tokens or [])
        
        for chunk in chunks:
            if chunk in special_set:
                pre_tokenized_list.append([chunk.encode("utf-8")])  
                continue
            
            for m in re.finditer(self.PAT2, chunk):
                byte_whole_token = m.group(0).encode("utf-8")
                bytes_seq =[bytes([x]) for x in byte_whole_token]
                print(bytes_seq)
                pre_tokenized_list.append(bytes_seq) 
        
        return  pre_tokenized_list