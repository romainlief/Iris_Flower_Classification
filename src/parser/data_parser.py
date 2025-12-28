from sklearn.datasets import load_iris
from pathlib import Path
from iris import Iris

class DataParser:
    def __init__(self, path: str) -> None:
        self.path = path
    
    def download(self):
        iris = load_iris(as_frame=True)
        df = iris.frame
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(exist_ok=True)
        csv_path = dataset_dir / "iris.csv"
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
    
    def parse(self):
        with open(self.path, 'r') as csvfile:
            data = csvfile.read()
        return data            