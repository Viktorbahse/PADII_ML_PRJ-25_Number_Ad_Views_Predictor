import pandas as pd
from typing import Dict, Any

class DummyModelFirst:
    def __init__(self):
        self.feature_columns = [
            'cpm', 'hour_start', 'hour_end', 
            'publishers', 'audience_size', 'user_ids'
        ]
        self.target_columns = [
            'at_least_one', 'at_least_two', 'at_least_three'
        ]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        missing_cols = [col for col in self.target_columns if col not in y.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")
        self.mean_at_least_one = y['at_least_one'].mean()
        self.mean_at_least_two = y['at_least_two'].mean()
        self.mean_at_least_three = y['at_least_three'].mean()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in self.feature_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")
        
        predictions = pd.DataFrame({
            'at_least_one': self.mean_at_least_one,
            'at_least_two': self.mean_at_least_two,
            'at_least_three': self.mean_at_least_three
        }, index=X.index)
        return predictions

class DummyModelSecond:
    def __init__(self):
        self.feature_columns = [
            'cpm', 'hour_start', 'hour_end', 
            'publishers', 'audience_size', 'user_ids'
        ]
        self.target_columns = [
            'at_least_one', 'at_least_two', 'at_least_three'
        ]
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        missing_cols = [col for col in self.feature_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")
        
        predictions = pd.DataFrame(
            0,  
            index=X.index,
            columns=self.target_columns
        )
        return predictions