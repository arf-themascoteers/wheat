from ann_simple import ANNSimple
from ds_manager import DSManager

dm = DSManager()
r2s = []

for fold_number, (train_ds, test_ds, validation_ds) in enumerate(dm.get_k_folds()):
    ann = ANNSimple(train_ds, test_ds, validation_ds)
    r2, rmse, pc = ann.run()
    print(f"r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")
    break




