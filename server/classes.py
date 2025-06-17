import pandas as pd
from typing import Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin

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

class MyPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2, model3, views_for_users_by_publishers, users_train, users_statistics_by_publishers):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.views_for_users_by_publishers = views_for_users_by_publishers
        self.users_train = users_train
        self.users_statistics_by_publishers = users_statistics_by_publishers

    def fit(self, X, y=None ):
        return self


    def count_views_by_user(self, user, publishers):
      sum = 0
      for publisher in publishers:
          sum += self.views_for_users_by_publishers[user][publisher]
      return sum

    def count(self, ans):
        sums = ans.groupby("user_id")["one"].sum()
        one = sums.mean()
        sums = ans.groupby("user_id")["two"].sum()
        two = sums.mean()
        sums = ans.groupby("user_id")["three"].sum()
        three = sums.mean()
        return one, two, three

    def predict(self, X):
        at_least_one = []
        at_least_two = []
        at_least_three = []
        for i in range(X.shape[0]):
            data, user_ids = self._preprocess(X.iloc[i])
            one = self.model1.predict(data)
            two = self.model2.predict(data)
            three = self.model3.predict(data)
            ans = pd.DataFrame({
                "user_id": user_ids,
                "one": one,
                "two": two,
                "three": three
            })
            one, two, three = self.count(ans)
            at_least_one.append(one)
            at_least_two.append(two)
            at_least_three.append(three)
        return pd.DataFrame({
            "at_least_one": at_least_one,
            "at_least_two": at_least_two,
            'at_least_three': at_least_three
        })

    def _preprocess(self, X):
        user_ids = []
        publishers = []
        cpm = []
        views_count_by_publisher = []
        duration = []
        total_views = []
        views_share = []
        min_cpm = []
        max_cpm = []
        mean_cpm = []
        median_cpm = []
        mean_views_per_day = []
        for user_id in list(map(int, (X['user_ids'].replace('(', '').replace('[', '').replace(']', '').replace(')', '').replace(',', ' ')).split())):
            sum_views = self.count_views_by_user(user_id, list(map(int, (X['publishers'].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', ' ')).split())))
            for publisher_id in list(map(int, (X['publishers'].replace('(', '').replace('[', '').replace(']', '').replace(')', '').replace(',', ' ')).split())):
                user_ids.append(user_id)
                publishers.append(publisher_id)
                duration.append(X['hour_end']-X['hour_start'])
                cpm.append(X['cpm'])
                total_views.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'total_views')])
                views_share.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'views_share')])
                min_cpm.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'min_cpm')])
                max_cpm.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'max_cpm')])
                mean_cpm.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'mean_cpm')])
                median_cpm.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'median_cpm')])
                mean_views_per_day.append(self.users_statistics_by_publishers[(user_id, publisher_id, 'mean_views_per_day')])
        data = pd.DataFrame({
            "user_id": user_ids,
            "publisher": publishers,
            "cpm": cpm,
            'total_view': total_views,
            'view_share': views_share,
            'min_cpm_per_publisher': min_cpm,
            'max_cpm_per_publisher': max_cpm,
            'mean_cpm_per_publisher': mean_cpm,
            'median_cpm_per_publisher': median_cpm,
            'mean_views_per_day': mean_views_per_day,
            'duration': duration
        })
        data = pd.merge(data, self.users_train, on='user_id', how='inner')
        data.fillna(0, inplace=True)
        return data, user_ids