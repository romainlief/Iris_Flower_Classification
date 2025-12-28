from sklearn.datasets import load_iris
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


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
        data = pd.read_csv(self.path)
        carac = data.drop(columns=['target']).values
        species = data['target'].values
        return carac, species 
    
    def split_data(self,features, labels, test_size=0.2, random_state=82): 
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test       