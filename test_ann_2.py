from ann_savi import ANNSAVI
from ds_manager import DSManager

dm = DSManager()
r2s = []

for fold_number, (train_ds, test_ds) in enumerate(dm.get_k_folds()):
    print(f"Train {len(train_ds)}; Test {len(test_ds)}")
    ann = ANNSAVI(train_ds, test_ds)
    r2, rmse, pc = ann.run()
    print(f"r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")
    r2s.append(r2)

print(r2s)
print(sum(r2s)/len(r2s))




