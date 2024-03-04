from ann_simple import ANNSimple
from ds_manager import DSManager

dm = DSManager()
r2s = []

for fold_number, (train_ds, test_ds) in enumerate(dm.get_k_folds()):
    print(f"Train {len(train_ds)}; Test {len(test_ds)}")
    ann = ANNSimple(train_ds, test_ds)
    r2, rmse, pc = ann.run()
    print(f"r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")
    break




