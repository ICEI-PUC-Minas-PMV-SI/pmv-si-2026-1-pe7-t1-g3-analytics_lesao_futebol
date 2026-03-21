import kagglehub
import os
import pandas as pd

def load_injuries_data():
  path = kagglehub.dataset_download("sananmuzaffarov/european-football-injuries-2020-2025")
  file_name = [f for f in os.listdir(path) if f.endswith(".csv")][0]
  file_path = os.path.join(path, file_name)

  df = pd.read_csv(file_path)
  return df