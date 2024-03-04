import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import utils
from sklearn import model_selection
from soil_dataset import SoilDataset


class DSManager:
    def __init__(self, folds=3, feature_set=None):
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
        df["SAVI"] = ((df["B8_convolved"] - df["B4_convolved"]) / (df["B8_convolved"] + df["B4_convolved"] + L)) * (1 + L)
        df["NDVI"] = ((df["B8_convolved"] - df["B4_convolved"]) / (df["B8_convolved"] + df["B4_convolved"]))
        #cols = ['B4_convolved','B8_convolved',"Ncontent"]
        cols = ['SAVI',"Ncontent"]
        df = df[cols]
        df = df.dropna()
        for col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
        df = df.sample(frac=1, random_state=1)
        self.data = df.to_numpy()

    def get_dss(self):
        train_data, test_data = model_selection.train_test_split(self.data, test_size=0.2, random_state=2)
        return SoilDataset(train_data[:,0:-1], train_data[:,-1]), SoilDataset(test_data[:,0:-1], test_data[:,-1])


if __name__ == "__main__":
    print("hi")

