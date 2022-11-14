import pickle

with open("agressive.pickle", "rb") as file:
    f = pickle.load(file)
print(f.keys())
print(
    f["state"].shape,
    f["action"].shape,
    f["reward"].shape,
    f["terminal"].shape,
    f["label"].shape,
)
