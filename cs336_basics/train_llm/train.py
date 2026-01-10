import argparse


def parse_args():
  paser = argparse.ArgumentParser(
    description="训练llm",
    formatter_class= argparse.ArgumentDefaultsHelpFormatter
  )

  paser.add_argument("--batch-size",type=int,default=32,help="每个批次的样本数")

  return paser.parse_args()
  
def load_datasets():
  pass
  
def build_model():
  pass

def build_optimizer():
  pass

def train_loop(
  model,
  optimizer,
  train_data,
  val_data,
):
  pass 

def main():
  train_data,val_data = load_datasets()
  model = build_model()
  optimizer = build_optimizer()
  
  train_loop(        
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data)
    


if __name__ == "__main__":
  args = parse_args()
  main(args)