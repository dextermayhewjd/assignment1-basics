def parse_loss(line: str) -> float:
    """
    Parse loss value from a log line like:
    '[valid] step   1200 | loss 2.3456789 | lr 1.00e-04 | t 456.7'
    """
    for part in line.split("|"):
        part = part.strip()
        if part.startswith("loss"):
            # "loss 2.3456789"
            return float(part.split()[1])
    raise ValueError(f"Cannot parse loss from line: {line}")
