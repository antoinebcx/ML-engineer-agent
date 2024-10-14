import pandas as pd
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, file_path, target_column, task_type):
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        self.task_type = task_type
        self.X_train, self.X_val, self.y_train, self.y_val = self._split_data()

    def _split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def get_data(self):
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'y_train': self.y_train,
            'y_val': self.y_val
        }

    def get_data_summary(self):
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'target': self.target_column,
            'task_type': self.task_type
        }