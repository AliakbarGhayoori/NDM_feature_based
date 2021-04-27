import numpy as np
import pickle

DATA_ADDRESS = "/media/external_3TB/3TB/ramezani/pmoini/NDM-feature/data/twitter-our"

with open(f"{DATA_ADDRESS}/cascade.txt", "r") as f:
    content = np.array(f.read().strip().split("\n"))
    np.random.shuffle(content)
    train_data = content[:50]

with open(f"{DATA_ADDRESS}/cascadetest.txt", "r") as f:
    content = np.array(f.read().strip().split("\n"))
    np.random.shuffle(content)
    test_data = content[:20]

with open(f"{DATA_ADDRESS}/cascadevalid.txt", "r") as f:
    content = np.array(f.read().strip().split("\n"))
    np.random.shuffle(content)
    valid_data = content[:20]

users = set()

for cascade in train_data:
    for ut in cascade.split(" "):
        users.add(int(ut.split(",")[0]))

for cascade in test_data:
    for ut in cascade.split(" "):
        users.add(int(ut.split(",")[0]))

for cascade in valid_data:
    for ut in cascade.split(" "):
        users.add(int(ut.split(",")[0]))


users_list = ['<blank>', '</s>'] + list(users)
u2idx = dict()
idx2u = users_list
idx2vec = dict()
vectors = pickle.load(open(f"{DATA_ADDRESS}/idx2vec.pickle", "rb"))
index_user = pickle.load(open(f"{DATA_ADDRESS}/u2idx.pickle", "rb"))


for i, u in enumerate(users_list):
    u2idx[u] = u
    if i <= 1:
        continue
    idx2vec[i] = vectors[index_user[str(u)]]

pickle.dump(u2idx, open("./data/u2idx.pickle", "wb"))
pickle.dump(idx2u, open("./data/idx2u.pickle", "wb"))
pickle.dump(idx2vec, open("./data/idx2vec.pickle", "wb"))

with open("data/cascade.txt", "w") as f:
    f.write("\n".join(train_data))

with open("data/cascadetest.txt", "w") as f:
    f.write("\n".join(test_data))

with open("data/cascadevalid.txt", "w") as f:
    f.write("\n".join(valid_data))


print("DONE")

