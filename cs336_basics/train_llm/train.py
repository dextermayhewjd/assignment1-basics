

import torch
import argparse
from pathlib import Path
import numpy as np
import time
import numpy as np
import torch
import numpy.typing as npt

from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from cs336_basics.config.all_in_all_config import Config
from pathlib import Path



from cs336_basics.transformer_assembling.transformer_language_model import Transformer_LM
from cs336_basics.training_module.adamw import AdamW
from cs336_basics.training_module.get_batch import get_batch
from cs336_basics.training_module.cross_entropy import cross_entropy
from cs336_basics.training_module.gradient_clipping import gradient_clipping
from cs336_basics.training_module.check_point import save_checkpoint,load_checkpoint
from cs336_basics.training_module.learning_rate_scheduling import learning_rate_scheduling
from cs336_basics.experiment_infra.loss_log import log_loss

from cs336_basics.training_module.check_point import save_checkpoint,load_checkpoint
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
    # # 读取data的path
    # parser.add_argument("--data-dir",type=Path,required=True)
    
    # '''
    # 需要二次确认一下
    # '''
    
    # # 模型的基本参数
    # parser.add_argument("--vocab-size",type=int,default=10_000,
    #                    help="词汇量 即tokenizer训练出来的 此处默认10_000")
    # parser.add_argument("--context-length",type=int,default=256,
    #                    help="tiny story并不需要long context")
    # parser.add_argument("--d-model",type=int,default=512,
    #                    help="d_model模型的维度 tinysory 是512")
    # parser.add_argument("--num-layers",type=int,default=4,
    #                    help="transformer 的层数")
    # parser.add_argument("--num-heads",type=int,default=16,
    #                    help="mha中 的head数量")
    # parser.add_argument("--d_ff",type=int,default=1344,
    #                    help="在ff 层的 要求是 8/3的d_model 但是要兼顾64的倍数来加速gpu")
    # parser.add_argument("--rope-theta",type = int,default=10_000,
    #                    help="rope的参数")
    
    # # learning rate parser
    # parser.add_argument("--lr-max", type=float, default=1e-4,
    #                     help="在learning rate scheduler中是 amax a是学习率")
    # parser.add_argument("--lr-min", type=float, default=1e-5,
    #                     help="learning rate scheduler 中是 amin")    
    # parser.add_argument("--warmup-steps",type=int,default=30_000,
    #                     help="T_c: step at which warm up ends")
    # parser.add_argument("--cosine-steps",type=int,default=100_000,
    #     help="T_c: step at which cosine decay ends 可以设计为这里结束 也可以做最后收敛")
    
    
    # # 优化器的参数 AdamW
    # parser.add_argument("--beta1", type=float, default=0.9,
    #                     help="这里控制m的平均步数 0.9 代表记住10步")
    # parser.add_argument("--beta2", type=float, default=0.95,
    #                     help="代表v是平均20步的平均振动")
    # parser.add_argument("--eps", type=float, default=1e-8,
    #                     help= "防止出现m/根号v平方 除0的情况")
    # parser.add_argument("--weight-decay", type=float, default=0.1,
    #                     help="一般是0.1到0.01之间")
    
    # # training
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--max-steps", type=int, default=100_000)
    # parser.add_argument("--seed",type=int,default=1337,
    #                     help="随机种子，保证实验可复现")
    # # checkpoint
    # parser.add_argument("--run-dir", type=Path,
    #                     default=Path("/home/fredkeira/data/cs336_training_result"),
    #                     help="checkpoint 存的folder在哪")
    # parser.add_argument("--save-every", type=int, default=500,
    #                     help="多久存一次")    
    # parser.add_argument("--eval-every", type=int, default=100,
    #                     help="多久valid一次")    
    # parser.add_argument("--exp-name",type=str, default="tinystories_base")
    
    
    

    # # 下面有些还没有实现但是长期一定有意义的参数
    # parser.add_argument("--device",type=str,default="cuda:1",
    #                     choices=["cuda:0", "cuda:1"],
    #                     help="训练设备两张卡选哪张")
    # parser.add_argument("--dtype",
    #                       type=str,
    #                       default="float32",
    #                       choices=["float32", "float16", "bfloat16"],
    #                       help="模型与计算精度")
    
    return parser.parse_args()

def make_run_dir(root: Path, exp_base_name: str, seed: int) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / exp_base_name / f"{ts}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# 传递一个dir path 读取 folder中的两个数据集，返回train_data, val_data
def load_datasets(path):
    
    train_bin = path / "train_ids.bin"
    val_bin = path / "valid_ids.bin"
    
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    return train_data, val_data


@torch.no_grad()
def evaluate(model, 
             valid_data: npt.NDArray, 
             cfg, device: str) -> float:
  
    model.eval()
    losses = []
    eval_steps = int(cfg.train.get("eval_steps", 1))  # 如果你没配 eval_steps，默认 1
    for _ in range(eval_steps):
        x, y = get_batch(
            dataset=valid_data,
            batch_size=int(cfg.train.batch_size),
            context_length=int(cfg.model.context_length),
            device=device,
        )
        logits = model(x)
        loss = cross_entropy(logits=logits, targets=y)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))



def train_loop(
                model,
                optimizer,
                train_data: npt.NDArray,
                valid_data: npt.NDArray,
                eval_every:int,
                batch_size: int,
                context_length: int,
                device: str,
                max_steps:int,
                lr_max,
                lr_min,
                T_w,
                T_c,
                max_l2_norm:float,
                run_dir:Path,
                exp_base_name:str,
                seed:int,
                save_every:int,
              ):
    start_time = time.time()
    model.to(device)
    model.train()
    step = 0
    # experiment dir 
    exp_dir = make_run_dir(
                root=run_dir,
                exp_base_name=exp_base_name,
                seed=seed,
                )
    
    ckpt_dir = exp_dir / "ckpt_dir"
    ckpt_dir.mkdir(parents=True,exist_ok=False)
    
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
        
        #后续可以升级为 不同层 不同学习率
        for group in optimizer.param_groups:
            group['lr'] = lr
        # ---------- 6. optimizer step ----------
        optimizer.step()
        
        # ---------- 7. logging ----------
        now = time.time()
        elps_time = now - start_time
        log_loss(
          tag="train",
          step= t,
          loss = loss,
          lr= lr,
          wallclock_time= elps_time,
          path= exp_dir,
          to_terminal=True
        )
        # ---------- 8. checkpoint ----------
        if t % eval_every ==0: 
            model.eval()
            x_valid,y_valid = get_batch(
                          dataset=valid_data,
                          batch_size= batch_size,
                          context_length=context_length,
                          device= device  
                        )
            with torch.no_grad():
                logits_valid = model(x_valid)
                
                valid_loss = cross_entropy(
                    logits=logits_valid,
                    targets=y_valid
                )
                now = time.time()
                elps_time = now - start_time
                
                log_loss(
                    tag="valid",
                    step= t,
                    loss = valid_loss,
                    lr= lr,
                    wallclock_time= elps_time,
                    path= exp_dir,
                    to_terminal=True
                      )
                model.train()
#         # ---------- 8. checkpoint ----------
        if t % save_every == 0:
            
            ckpt_dir_file = ckpt_dir/f"step{t}"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=t,
                out= ckpt_dir_file
            )
        
def main(*args):
    args = parse_args()
    ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
    OWT_DATA_REPO = ASSIGNMENT_REPO /"token_to_id_outputs" 

    config_path = Path("/home/fredkeira/projects/assignment1-basics/cs336_basics/config/config.yaml")

    base_cfg = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, file_cfg)
    
    device = args
    
    train_data,val_data = load_datasets(path=OWT_DATA_REPO)
    
    model = Transformer_LM(
                            vocab_size=args.v,
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
    
    train_loop(model=model,
               device= device,
               max_steps= args.,
               train_data= train_data,
               valid_data = val_data,
               context_length=args,
               optimizer= optimizer,
               max_l2_norm= args.,
               lr_max=args,
               lr_min=args.,
               T_w=args.,
               T_c=args.,
               )
    


if __name__ == "__main__":
  args = parse_args()
  main(args)