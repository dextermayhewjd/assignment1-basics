

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
    paser = argparse.ArgumentParser(
      description="训练llm",
      formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )

    paser.add_argument("--batch-size",type=int,default=32,help="每个批次的样本数")

    return paser.parse_args()

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


def train_loop(
                model,
                optimizer,
                train_data,
                batch_size: int,
                context_length: int,
                device: str,
                max_steps,
                lr_max,
                lr_min,
                T_w,
                T_c,
                max_l2_norm,
                ckpt_path,
                save_every
              ):

    model.to(device)
    model.train()

    
    
    step = 0
    
    for t in range(step, max_steps):
        # ---------- 1. sample batch ----------
        x ,y = get_batch(
                          dataset=train_data,
                          batch_size= batch_size,
                          context_length=context_length,
                          device= device  
                        )
        # ---------- 2. forward----------
        logits = model(x)
        loss = cross_entropy(logits=logits,
                             targets=y
                             )
        # ---------- 3. backward ----------
        optimizer.zero_grad()
        loss.backward()
        
    

        # ---------- 4. gradient clipping ----------
        gradient_clipping(
                          params=model.parameters(),
                          max_l2_norm= max_l2_norm
                          )
        # ---------- 5. learning rate scheduling ----------
        lr = learning_rate_scheduling(
            t=t,
            a_max=lr_max,
            a_min=lr_min,
            T_w=T_w,
            T_c=T_c,
        )
        for group in optimizer.param_groups:
            group['lr'] = lr
# ---------- 6. optimizer step ----------
        optimizer.step()

        # ---------- 7. logging ----------
        if t % 100 == 0:
            print(f"step {t} | loss {loss.item():.4f} | lr {lr:.2e}")

        # ---------- 8. checkpoint ----------
        if t % save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                iteration=t,
                out=ckpt_path
            )
        
def main():
  
    ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
    OWT_DATA_REPO = ASSIGNMENT_REPO /"token_to_id_outputs" 
    # 这个vocab size是基于owt数据集训练的 BPE tokenizer 得到的
    VOCAB_SIZE = 50257
    # 模型超参数 上下文长度
    CONTEXT_LENGTH = 2048
    # 模型的dimension
    D_MODEL = 768
    # 多少层 transformer
    NUM_LAYERS = 12
    NUM_HEADS = 12
    # feedford 层
    D_FF = 3072
    # 旋转位置编码的 theta 参数
    ROPE_THETA = 1000000.0
    
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
    
    train_loop(        
              model=model,
              optimizer=optimizer,
              train_data=train_data,
              val_data=val_data
              batch_size=32,
              context_length=CONTEXT_LENGTH,
              device="cuda",
              max_steps=100000,
              lr_max=1e-4,
              lr_min=1e-5,
              T_w=1000,
              T_c=100000,
              max_l2_norm=1.0,
              ckpt_path= ASSIGNMENT_REPO,
              save_every=1000
              )
    


if __name__ == "__main__":
  args = parse_args()
  main(args)