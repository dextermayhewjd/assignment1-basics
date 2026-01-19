

import torch
import argparse
from pathlib import Path
import numpy as np
import time
import numpy as np
import torch
import numpy.typing as npt
import random
import torch
import numpy as np

from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from cs336_basics.config.all_in_all_config import Config
from pathlib import Path
from typing import cast

# 临时替换以排除自定义函数 bug
import torch.nn.functional as F

from cs336_basics.transformer_assembling.transformer_language_model import Transformer_LM
from cs336_basics.training_module.adamw import AdamW
from cs336_basics.training_module.get_batch import get_batch
from cs336_basics.training_module.cross_entropy import cross_entropy
from cs336_basics.training_module.gradient_clipping import gradient_clipping
from cs336_basics.training_module.check_point import save_checkpoint,load_checkpoint
from cs336_basics.training_module.learning_rate_scheduling import learning_rate_scheduling
from cs336_basics.experiment_infra.loss_log import log_loss
from cs336_basics.experiment_infra.load_best_valid_from_log import load_best_valid_from_log
from cs336_basics.training_module.check_point import save_checkpoint,load_checkpoint

from cs336_basics.config.all_in_all_config import DataConfig,MemmapDType
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
def single_step_loss(logits, y):
    # logits: (1, T, V)
    # y:      (1, T)
    logits_last = logits[:, -1, :]   # (1, V)
    y_last = y[:, -1]                # (1,)
    return cross_entropy(logits_last, y_last)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # 为了可复现（会稍慢）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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
# def load_datasets(path):
    
#     train_bin = path / "train_ids.bin"
#     val_bin = path / "valid_ids.bin"
    
#     train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
#     val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
#     return train_data, val_data

def load_datasets(cfg: DataConfig):
    data_dir = cfg.data_dir
    train_bin = data_dir / cfg.train_bin_name
    val_bin = data_dir / cfg.valid_bin_name

    dtype = np.uint16 if cfg.memmap_dtype == MemmapDType.uint16 else np.uint32

    train_data = np.memmap(train_bin, dtype=dtype, mode="r")
    val_data = np.memmap(val_bin, dtype=dtype, mode="r")
    return train_data, val_data


@torch.no_grad()
def evaluate(
            model,
            dataset,
            batch_size,
            context_length,
            device,
            eval_steps: int,
    ):
        assert not torch.is_grad_enabled()
        model.eval()
        losses = []

        for _ in range(eval_steps):
            x, y = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(loss.item())

        model.train()
        return sum(losses) / len(losses)




def train_loop(
                cfg,
                model,
                optimizer,
                train_data: npt.NDArray,
                valid_data: npt.NDArray,
                
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
                eval_every:int,
                eval_step:int,
                resume:bool,
                resume_dir:Path|None,
                debug_overfit:bool = False,
                one_step:bool = False,
                debug_shapes:bool = False,
                debug_norms:bool = False
              ):
    start_time = time.time()
    model.to(device)
    model.train()
    
    # ===== debug: 固定一个 batch =====

    if debug_overfit:
        # 这里 get_batch 已经返回了 x: batch[:, :-1] 和 y: batch[:, 1:]
        x_fixed, y_fixed = get_batch(
            train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        # 调试打印：确认 x 的最后一个 token 确实是 y 倒数第二个 token
        print(f"DEBUG: x_fixed shape {x_fixed.shape}, y_fixed shape {y_fixed.shape}")
    
    if resume:
        assert resume_dir is not None, "resume=True but resume_dir is None"
        exp_dir = resume_dir

        ckpt_dir = exp_dir / "ckpt_dir"
        # 工程正确 确保就算folder也能继续
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = exp_dir / "ckpt_latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing ckpt_latest.pt under {exp_dir}")

        start_step = load_checkpoint(
                        ckpt_path,
                        model=model,
                        optimizer=optimizer,
                    ) + 1
        best_valid_loss = load_best_valid_from_log(exp_dir) \
            if (exp_dir / "valid.log").exists() else float("inf")
    
    else:
        start_step = 0
        # experiment dir 
        exp_dir = make_run_dir(
                    root=run_dir,
                    exp_base_name=exp_base_name,
                    seed=seed,
                    )
        OmegaConf.save(cfg, exp_dir/"config.yaml")
        
        ckpt_dir = exp_dir / "ckpt_dir"
        ckpt_dir.mkdir(parents=True,exist_ok=False)

        best_valid_loss = float("inf")
        
    for step in range(start_step, max_steps):
        # ---------- 1. sample batch ----------
        if debug_overfit:
            x, y = x_fixed, y_fixed
        else:
            x ,y = get_batch(
                            dataset=train_data,
                            batch_size= batch_size,
                            context_length=context_length,
                            device= device  
                            )
        # ---------- 2. forward ----------
        logits = model(x)  # logits shape: (B, T, V)
        
        if one_step:
            # 单步模式只算最后一个 token 的 loss
            loss = single_step_loss(logits, y)
            
        else:
            # 正常模式算全序列 loss
            # ！！！注意：因为 get_batch 已经把 y 偏移好了，
            # 这里的 logits[:, t] 对应的就是 y[:, t] (即下一个词)
            # 所以直接传入即可，不需要在 cross_entropy 内部再 shift
            loss = cross_entropy(logits=logits, targets=y)
            # 官方用于测试loss function 有没有写对 写对了 
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
        # ---------- 3. backward ----------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # ---------- 调试：监控梯度范数 (重要！) ----------
        if debug_norms and step % 10 == 0:
            total_grad_norm = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Step {step} | Loss: {loss.item():.4f} | Total Grad Norm: {total_grad_norm:.4f}")
        
        
        # ---------- 4. gradient clipping ----------
        gradient_clipping(
                          params=model.parameters(),
                          max_l2_norm= max_l2_norm
                          )
        # ---------- 5. learning rate scheduling ----------
        # 这个是global lr
        lr = learning_rate_scheduling(
            t=step,
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
          step= step,
          loss = loss,
          lr= lr,
          wallclock_time= elps_time,
          run_dir= exp_dir,
          to_terminal=True
        )
        
        # ---------- 8. checkpoint ----------

            
        if step % eval_every ==0: 
            valid_loss = evaluate(
                model=model,
                dataset=valid_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                eval_steps=eval_step
            )
            
            
            elps_time = now - start_time
            log_loss(tag="valid",
                     step=step,
                     loss=valid_loss,
                     lr=lr,
                     wallclock_time=elps_time,
                     run_dir=exp_dir,
                     to_terminal=True,
                     )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                ckpt_best_file = exp_dir/f"ckpt_best.pt"
                save_checkpoint(model=model,optimizer=optimizer,
                iteration=step,out= ckpt_best_file)
#         # ---------- 8. checkpoint ----------
        if step % save_every == 0:
            
            ckpt_dir_file = ckpt_dir/f"step{step}.pt"
            # 存放最新的latest file
            ckpt_latest_file = exp_dir/f"ckpt_latest.pt"
            
            # 存当前的checkpoint
            save_checkpoint(model=model,optimizer=optimizer,
                            iteration=step,out= ckpt_dir_file)
            save_checkpoint(model=model,optimizer=optimizer,
                iteration=step,out= ckpt_latest_file)
        
def main():
    args = parse_args()
    ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
    OWT_DATA_REPO = ASSIGNMENT_REPO /"token_to_id_outputs" 

    config_path = Path("/home/fredkeira/projects/assignment1-basics/cs336_basics/config/debug_config.yaml")

    base_cfg = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, file_cfg)
    cfg = cast(Config, cfg)
    # set_seed(cfg.train.seed)
    
    # train_data,val_data = load_datasets(path=OWT_DATA_REPO)
    train_data, val_data = load_datasets(cfg.data)

    
    model = Transformer_LM(
                            vocab_size=cfg.model.vocab_size,
                            context_length=cfg.model.context_length,
                            d_model=cfg.model.d_model,
                            num_layers=cfg.model.num_layers,
                            num_heads=cfg.model.num_heads,
                            d_ff=cfg.model.d_ff,
                            rope_theta=cfg.model.rope_theta,
                          )
    # 其实这里日后会更具learning rate scheduling 变更
    optimizer = AdamW(
                        params=model.parameters(),
                        lr=cfg.optim.lr_max,
                        weight_decay=cfg.optim.weight_decay,
                        betas=(cfg.optim.beta1,cfg.optim.beta2),
                        eps=(cfg.optim.eps)
                      )
    
    
    train_loop(
                cfg=cfg,
               model=model,
               optimizer= optimizer,
               train_data= train_data,
               valid_data = val_data,
               batch_size= cfg.train.batch_size,
               context_length=cfg.model.context_length,
               device= cfg.runtime.device,
               max_steps= cfg.train.max_steps,
               lr_max=cfg.optim.lr_max,
               lr_min=cfg.optim.lr_min,
               T_w=cfg.optim.warmup_steps,
               T_c=cfg.optim.cosine_steps,               
               max_l2_norm= cfg.train.max_l2_norm,
               run_dir=cfg.experiment.run_dir,
               exp_base_name= cfg.experiment.exp_name,
               seed= cfg.train.seed,
               save_every=cfg.train.save_every,
               eval_every=cfg.train.eval_every,
               eval_step=cfg.train.eval_steps,
               resume=cfg.experiment.resume,
               resume_dir=cfg.experiment.resume_path,
                debug_overfit =cfg.experiment.debug_overfit ,
                one_step=cfg.experiment.one_step,
                debug_shapes=cfg.experiment.debug_shapes,
                debug_norms=cfg.experiment.debug_norms,
               )
    


if __name__ == "__main__":
  main()