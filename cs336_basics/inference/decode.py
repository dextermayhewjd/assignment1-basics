from jaxtyping import Int ,Float
import torch
from cs336_basics.transformer_assembling.transformer_language_model import Transformer_LM
from cs336_basics.final_solutions.tokenizer2 import Tokenizer
from cs336_basics.training_module.top_p_next_token import top_p_sample_next_token
# naive autoregressive decoding (no KV cache)
def decode(
    model: Transformer_LM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """
    Naive autoregressive decoding (no KV cache).

    Args:
        model: trained Transformer language model
        tokenizer: tokenizer used for encoding / decoding
        prompt: input text prompt
        max_new_tokens: maximum number of tokens to generate
        temperature: softmax temperature
        top_p: nucleus sampling threshold
    """
    
    
    '''


    '''
    model.eval()
    device = next(model.parameters()).device

    # 1. Encode prompt
    generated_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device
    )

    new_token_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            
            # ids 是list[int] 变成tensor后 然后再加了一个batch 然后放进model
            # 2. Forward pass on current sequence
            logits = model(generated_ids.unsqueeze(0))  # (1, T, V)

            # 3. Get logits for next token
            next_token_logits = logits[:, -1, :]        # (1, V)

            # 4. Sample next token id 这里返回的是 0dim scalar
            next_token = top_p_sample_next_token(
                input_features=next_token_logits,
                dimension=-1,
                temp=temperature,
                p=top_p
            ).item()

            # 5. Append token
            new_token_ids.append(next_token)

            #把 scalar 变成0-dim tensor 
            next_token_tensor = torch.tensor(
                [next_token],
                device=device,
                dtype=torch.long
            )
            generated_ids = torch.cat(
                [generated_ids, next_token_tensor],
                dim=0
            )

    # 6. Decode
    return (
        tokenizer.decode(generated_ids.tolist()),
        tokenizer.decode(new_token_ids),
    )
