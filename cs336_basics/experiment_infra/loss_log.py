# from pathlib import Path
# def train_loss_log(
#                     loss:float,
#                     to_terminal:bool,
#                     step:int,
#                     lr:float,
#                     use_wandb:bool,
#                     wallclock_time: float,
#                     path:Path  
                    
#                 ):
#     path.parent.mkdir(parents=True,exist_ok=True)
#     with open(path,"a") as f:
#         f.write(
#             f"step{step:6d} | "
#             f"loss{loss:10.7f}|"
#             f"lr{lr:10.2e}|"
#             f"t{wallclock_time:10.1f}"
#         )
    
#     if to_terminal:
#         print(
#             f"step{step:6d} | "
#             f"loss{loss:10.7f}|"
#             f"lr{lr:10.2e}|"
#             f"t{wallclock_time:10.1f}"
#             )
    
    
    
# def valid_loss_log(
#                     loss:float,
#                     to_terminal:bool,
#                     step:int,
#                     lr:float,
#                     use_wandb:bool,
#                     wallclock_time: float,
#                     path:Path  
                    
#                 ):
#     path.parent.mkdir(parents=True,exist_ok=True)
#     with open(path,"a") as f:
#         f.write(
#             f"step{step:6d} | "
#             f"loss{loss:10.7f}|"
#             f"lr{lr:10.2e}|"
#             f"t{wallclock_time:10.1f}\n"
#         )
    
#     if to_terminal:
#         print(
#             f"step{step:6d} | "
#             f"loss{loss:10.7f}|"
#             f"lr{lr:10.2e}|"
#             f"t{wallclock_time:10.1f}"
#             )

from pathlib import Path
import wandb

def log_loss(
    *,
    tag: str,            # "train" / "valid"
    step: int,
    loss,
    lr: float,
    wallclock_time: float,
    run_dir: Path,
    to_terminal: bool = True,
):
    # 明确：run_dir 是目录
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"{tag}.log"
    loss_val = float(loss)
    
    line = (
        f"[{tag:5}] "
        f"step{step:6d} | "
        f"loss{loss_val:10.7f} | "
        f"lr{lr:10.2e} | "
        f"t{wallclock_time:10.1f}"
    )
    # 2. wandb logging（新增）
    if wandb.run is not None:
        wandb.log(
            {
                f"{tag}/loss": float(loss),
                f"{tag}/lr": lr,
                f"{tag}/wallclock_time": wallclock_time,
            },
            step=step,
        )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if to_terminal:
        print(line)