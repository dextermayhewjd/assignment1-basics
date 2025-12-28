import pickle
import regex as re 
from collections.abc import Iterable,Iterator
PAT2 = re.compile(r"""
    '(?:[sdmt]|ll|ve|re)
    | \ ?\p{L}+
    | \ ?\p{N}+
    | \ ?[^\s\p{L}\p{N}]+
    | \s+(?!\S)
    | \s+
    """, re.VERBOSE)

class Tokenizer:

    
    def __init__(self,
                 vocab:dict[int,bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.special_tokens = sorted(self.special_tokens or [], key=len, reverse=True)

        self.bytes2int:dict[bytes,int] = {v:k for k,v in self.vocab.items()}
        
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            assert b in self.bytes2int, (
                f"Special token {tok} not found in vocab. "
                "Did you forget to add it during BPE training?"
        )

                
            
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

    def encode(self, text: str)-> list[int]:
        ids = []
        for bytes_seq in self._pre_tokenize(text=text):
            if isinstance(bytes_seq, tuple) and bytes_seq[0] == "SPECIAL":
                ids.append(self.bytes2int[bytes_seq[1].encode("utf-8")])
            else:
                ids.extend(self._encode_one_token(bytes_seq))
        return ids

    def _encode_one_token(self,token:list[bytes])->list[int]:
        # print("INPUT:", token)

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
        
    
           
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

            
    def decode(self, ids:list[int])-> str:
        bytes_seq = b''
        for single_int in ids:
            i_b = self.vocab.get(single_int)
            if i_b is None:
                raise KeyError(f"Unknown token id: {i_b}")
            bytes_seq += i_b
        return bytes_seq.decode("utf-8", errors="replace")      
    
    #"You are cat<|speical|>" -> ["You are cat","<|speical|>"]
    def split_on_complete_specials(self,
                                 text: str
                                 ) -> list[str]:
        """
        Split text on COMPLETE special tokens only.
        Incomplete prefixes are treated as normal text.
        """
        if not self.special_tokens:
            return [text]
        # pattern = "|".join(re.escape(tok) for tok in special_tokens)
        #关键：加 () 捕获组，让 special token 也出现在 split 结果里
        pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        
        return [p for p in re.split(pattern, text) if p]
    
    def split_with_special_prefix(self, text: str):
        special_tokens = self.special_tokens or []
        if not special_tokens:
            return [text], ""

        # 1. 先切完整 special token（这是安全的）
        pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        parts = [p for p in re.split(pattern, text) if p]

        if not parts:
            return [], ""

        tail = parts[-1]

        # 2. 在 tail 中找“最长 suffix”，使其是 special token 的前缀
        max_prefix_len = max(len(tok) for tok in special_tokens)

        for i in range(1, min(len(tail), max_prefix_len) + 1):
            suffix = tail[-i:]
            if any(tok.startswith(suffix) for tok in special_tokens):
                # 命中：tail 的一部分可能是 special token 前缀
                safe_head = tail[:-i]
                new_parts = parts[:-1]
                if safe_head:
                    new_parts.append(safe_head)
                return new_parts, suffix

        # 3. 没有任何前缀可能
        return parts, ""

    
    
    
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
        
        chunks = self.split_on_complete_specials(text)
        # print(chunks)
        special_set = set(self.special_tokens or [])
        
        for chunk in chunks:
            if chunk in special_set:
                pre_tokenized_list.append(("SPECIAL", chunk))  
                continue
            
            for m in re.finditer(self.PAT2, chunk):
                byte_whole_token = m.group(0).encode("utf-8")
                bytes_seq =[bytes([x]) for x in byte_whole_token]
                # print(bytes_seq)
                pre_tokenized_list.append(bytes_seq) 
        
        return  pre_tokenized_list
    
