import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import utils
from soil_dataset import SoilDataset


class DSManager:
    def __init__(self, folds=10, feature_set=None):        
        self.folds = folds
        self.feature_set = feature_set
        
        if self.feature_set is None:
            self.feature_set = utils.get_all_features()
        
        torch.manual_seed(0)
        
        df = pd.read_csv(utils.get_data_file())
        cols = ['B1_convolved', 'B2_convolved', 'B3_convolved', 'B4_convolved', 'B5_convolved', 'B6_convolved',
                'B7_convolved', 'B8_convolved', 'B8A_convolved', 'B9_convolved', 'B11_convolved', 'Ncontent']
        df = df[cols]

        L = 0.5
        df["SAVI"] = ((df["B8_convolved"] - df["B4_convolved"]) / (df["B8_convolved"] + df["B4_convolved"] + L)) * (
                    1 + L)
        df["NDVI"] = ((df["B8_convolved"] - df["B4_convolved"]) / (df["B8_convolved"] + df["B4_convolved"]))
        cols = ['SAVI', 'Ncontent']
        df = df[cols]
        df = df[cols].dropna()

        self.derived_columns = []
        for col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
        self.data = df.sample(frac=1).to_numpy()

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.data)):
            train_data = self.data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.data[test_index]
            train_x = train_data[:, 0:-1]
            train_y = train_data[:, -1]
            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]
            validation_x = validation_data[:, 0:-1]
            validation_y = validation_data[:, -1]

            yield SoilDataset(train_x, train_y), \
                SoilDataset(test_x, test_y), \
                SoilDataset(validation_x, validation_y)

    def get_folds(self):
        return self.folds


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from soil_dataset import SoilDataset
    dm = DSManager(3,["ndvi","b1","b4"])
    for fold_number, (dtrain, dtest, dval) in enumerate(dm.get_k_folds()):
        dataloader = DataLoader(dtrain, batch_size=500, shuffle=True)
        for batch_number, (x, y) in enumerate(dataloader):
            print(x.shape)
            print(y.shape)
            break
        break


