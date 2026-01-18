from pathlib import Path
from cs336_basics.experiment_infra.parse_loss import parse_loss
def load_best_valid_from_log(exp_dir: Path) -> float:
    log_path = exp_dir / "valid.log"
    if not log_path.exists():
        return float("inf")

    best = float("inf")
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                loss = parse_loss(line)
                best = min(best, loss)
            except Exception:
                # 跳过异常行（比如空行 / 格式变化）
                continue
    return best
