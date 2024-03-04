from ann_savi import ANNSAVI
from ds_manager import DSManager

dm = DSManager()
r2s = []

train_ds, test_ds = dm.get_dss()
print(f"Train {len(train_ds)}; Test {len(test_ds)}")
ann = ANNSAVI(train_ds, test_ds)
r2, rmse, pc = ann.run()
print(f"r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")





