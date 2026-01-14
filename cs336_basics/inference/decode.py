from jaxtyping import Int
import torch
# 这个版本的docode只适合prefill阶段 因为没有缓存机制
def decode(
            model,
            tokenizer,
            prompt_ids : Int[torch.Tensor, "Batch SeqLen"],
            max_new_tokens: Int,
            temperature: float,
            top_p
            ):  
    model.eval()
    ids = prompt_ids[:, :].to(model.device)
    
    for _ in range(max_new_tokens):
        # logits shape: (B, SeqLen, VocabSize) 最后一层LINEAR 将 d_model 映射到 vocab_size
        logits = model(ids)
        # 取最后一个 token 的 logits
        logits = logits[:,-1,:]
        
        