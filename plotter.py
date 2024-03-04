import numpy as np
import matplotlib.pyplot as plt
from ds_manager import DSManager


def plot_me(ds):
    sorted_indices = np.argsort(ds.x[:,1])
    x = ds.x[sorted_indices,1]
    y = ds.y[sorted_indices]
    plt.plot(x,y)
    plt.show()


if __name__ == "__main__":
    dm = DSManager()
    r2s = []

    for fold_number, (train_ds, test_ds) in enumerate(dm.get_k_folds()):
        print(f"Train {len(train_ds)}; Test {len(test_ds)}")
        plot_me(train_ds)
        plot_me(test_ds)
