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

def log_loss(
    *,
    tag: str,            # "train" / "valid"
    step: int,
    loss: float,
    lr: float,
    wallclock_time: float,
    path: Path,
    to_terminal: bool = True,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    line = (
        f"[{tag:5}] "
        f"step{step:6d} | "
        f"loss{loss:10.7f} | "
        f"lr{lr:10.2e} | "
        f"t{wallclock_time:10.1f}"
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if to_terminal:
        print(line)