import json
with open("GNN_data.json", "r", encoding='utf-8') as json_file:
    data_memory = json.load(json_file)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
X = [item[2] for item in data_memory]
y = [item[3] for item in data_memory]
# y_over = list()
# empty = list()
# for i in range(len(data_memory)):
#     item = data_memory[i]
#     if item[3] >= 5:
#         empty.append(i)
#         y_over.append(item[3])



tsne = TSNE(n_components=2, random_state=0, perplexity = 30)
X_2d = tsne.fit_transform(X)
X = X_2d.tolist()
with open("X2D_pol.json", "w", encoding='utf-8') as json_file:
    json.dump(X, json_file, ensure_ascii=False)

with open("y.json", "w", encoding='utf-8') as json_file:
    json.dump(y, json_file, ensure_ascii=False)

# X_over = np.array([X_2d[e] for e in empty])
#
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='rainbow', s = 0.5)
# scatter2 = plt.scatter(X_over[:, 0], X_over[:, 1], c='red', s = 2)
# cbar = plt.colorbar(scatter)
#
# plt.xlabel('hi1')
# plt.ylabel('hi2')
# plt.title('t-SNE visualization')
# plt.show()