from ann_savi import ANNSAVI
from ds_manager import DSManager

dm = DSManager()
r2s_normal = []
r2s_learnable = []

for i in range(10):
    train_ds, test_ds = dm.get_dss()

    ann = ANNSAVI(train_ds, test_ds)
    r2, rmse, pc = ann.run()
    print(f"Normal: r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")
    r2s_normal.append(r2)

    ann = ANNSAVI(train_ds, test_ds)
    ann.L.requires_grad = True
    r2, rmse, pc = ann.run()
    print(f"Learnable: r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")
    r2s_learnable.append(r2)


print("Normal",f"{sum(r2s_normal)/len(r2s_normal)}")
print("Learnable",f"{sum(r2s_learnable)/len(r2s_learnable)}")



