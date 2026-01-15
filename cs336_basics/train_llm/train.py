

import torch
import argparse
from pathlib import Path
import numpy as np
from cs336_basics.transformer_assembling.transformer_language_model import Transformer_LM
from cs336_basics.training_module.adamw import AdamW
from cs336_basics.training_module.get_batch import get_batch
from cs336_basics.training_module.cross_entropy import cross_entropy
from cs336_basics.training_module.gradient_clipping import gradient_clipping
from cs336_basics.training_module.check_point import save_checkpoint,load_checkpoint
from cs336_basics.training_module.learning_rate_scheduling import learning_rate_scheduling

'''
Docstring for cs336_basics.training_module.training_together

Tokenizer → token ids 离线化储存为bin文件
np.memmap  将文件和array映射起来
get_batch → (x, y) 需要传入的np array
TransformerLM → logits
cross_entropy → loss
AdamW → step
lr scheduler
gradient clipping
checkpoint save / load

'''


def parse_args():
    parser = argparse.ArgumentParser(
      description="训练llm",
      formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )
    # 读取data的path
    parser.add_argument("--data-dir",type=Path,required=True)
    
    '''
    需要二次确认一下
    '''
    
    # 模型的基本参数
    parser.add_argument("--vocab-size",type=int,default=10_000,
                       help="词汇量 即tokenizer训练出来的 此处默认10_000")
    parser.add_argument("--context_length",type=int,default=256,
                       help="tiny story并不需要long context")
    parser.add_argument("--d-model",type=int,default=512,
                       help="d_model模型的维度 tinysory 是512")
    parser.add_argument("--num-layers",type=int,default=4,
                       help="transformer 的层数")
    parser.add_argument("--num-heads",type=int,default=16,
                       help="mha中 的head数量")
    parser.add_argument("--d_ff",type=int,default=1344,
                       help="在ff 层的 要求是 8/3的d_model 但是要兼顾64的倍数来加速gpu")
    parser.add_argument("--rope-theta",type = int,default=10_000,
                       help="rope的参数")
    
    
    # 优化器的参数 AdamW
    parser.add_argument("--lr-max", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    
    # training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr-max", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # checkpoint
    
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


# 传递一个dir path 读取 folder中的两个数据集，返回train_data, val_data
def load_datasets(path):
    
    train_bin = path / "train_ids.bin"
    val_bin = path / "valid_ids.bin"
    
    train_data = np.memmap(
      train_bin,
      dtype=np.uint16,
      mode="r"
    )
    
    val_data = np.memmap(
      val_bin,
      dtype=np.uint16,
      mode="r"
    )
    return train_data, val_data


# def train_loop(
#                 model,
#                 optimizer,
#                 train_data,
#                 batch_size: int,
#                 context_length: int,
#                 device: str,
#                 max_steps,
#                 lr_max,
#                 lr_min,
#                 T_w,
#                 T_c,
#                 max_l2_norm,
#                 ckpt_path,
#                 save_every
#               ):

#     model.to(device)
#     model.train()

    
    
#     step = 0
    
#     for t in range(step, max_steps):
#         # ---------- 1. sample batch ----------
#         x ,y = get_batch(
#                           dataset=train_data,
#                           batch_size= batch_size,
#                           context_length=context_length,
#                           device= device  
#                         )
#         # ---------- 2. forward----------
#         logits = model(x)
#         loss = cross_entropy(logits=logits,
#                              targets=y
#                              )
#         # ---------- 3. backward ----------
#         optimizer.zero_grad()
#         loss.backward()
        
    

#         # ---------- 4. gradient clipping ----------
#         gradient_clipping(
#                           params=model.parameters(),
#                           max_l2_norm= max_l2_norm
#                           )
#         # ---------- 5. learning rate scheduling ----------
#         lr = learning_rate_scheduling(
#             t=t,
#             a_max=lr_max,
#             a_min=lr_min,
#             T_w=T_w,
#             T_c=T_c,
#         )
#         for group in optimizer.param_groups:
#             group['lr'] = lr
# # ---------- 6. optimizer step ----------
#         optimizer.step()

#         # ---------- 7. logging ----------
#         if t % 100 == 0:
#             print(f"step {t} | loss {loss.item():.4f} | lr {lr:.2e}")

#         # ---------- 8. checkpoint ----------
#         if t % save_every == 0:
#             save_checkpoint(
#                 model,
#                 optimizer,
#                 iteration=t,
#                 out=ckpt_path
#             )
        
def main():
  
    ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
    OWT_DATA_REPO = ASSIGNMENT_REPO /"token_to_id_outputs" 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_data,val_data = load_datasets(path=OWT_DATA_REPO)
    
    model = Transformer_LM(
                            vocab_size=VOCAB_SIZE,
                            context_length=CONTEXT_LENGTH,
                            d_model=D_MODEL,
                            num_layers=NUM_LAYERS,
                            num_heads=NUM_HEADS,
                            d_ff=D_FF,
                            rope_theta=ROPE_THETA,
                          )
    
    optimizer = AdamW(
                        params=model.parameters(),
                        lr=1e-4,
                        weight_decay=1e-2,
                      )
    

    


if __name__ == "__main__":
  args = parse_args()
  main(args)