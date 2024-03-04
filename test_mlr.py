from ds_manager import DSManager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


dm = DSManager()
r2s = []

for fold_number, (train_ds, test_ds) in enumerate(dm.get_k_folds()):
    print(f"Train {len(train_ds)}; Test {len(test_ds)}")

    train_x = train_ds.x
    train_y = train_ds.y
    test_x = test_ds.x
    test_y = test_ds.y
    model_instance = LinearRegression()
    model_instance = model_instance.fit(train_x, train_y)
    r2 = model_instance.score(test_x, test_y)
    print(f"r2={r2:.3f}")
    r2s.append(r2)

print(r2s)
print(sum(r2s)/len(r2s))




