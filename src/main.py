from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path

# Charger Iris depuis sklearn
iris = load_iris(as_frame=True)
df = iris.frame

dataset_dir = Path("dataset")
dataset_dir.mkdir(exist_ok=True)

csv_path = dataset_dir / "iris.csv"

df.to_csv(csv_path, index=False)