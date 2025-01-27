import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# 범위를 사용자 정의 범위로 설정
norm = colors.Normalize(vmin=-0.5, vmax=4.5)

# 데이터 로드
with open("X2D_raw.json", "r", encoding='utf-8') as json_file:
    data_memory = json.load(json_file)

with open("X2D_gnn.json", "r", encoding='utf-8') as json_file:
    data_memory1 = json.load(json_file)

with open("X2D_pol.json", "r", encoding='utf-8') as json_file:
    data_memory2 = json.load(json_file)

with open("y.json", "r", encoding='utf-8') as json_file:
    y = json.load(json_file)

y = np.array(y)
X_2d = np.array(data_memory)
X_2d1 = np.array(data_memory1)
X_2d2 = np.array(data_memory2)

# 범위를 사용자 정의 범위로 설정 및 색상 맵 설정
norm = colors.Normalize(vmin=-0.5, vmax=4.5)
cmap = 'turbo'
plt.rcParams['savefig.pad_inches'] = 0.6
# 첫 번째 그래프 생성 및 저장
plt.figure(figsize=(6, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, norm=norm, s=2)
plt.xticks([])
plt.yticks([])
#plt.colorbar(label='Estimation of Action State Value', orientation='vertical')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.savefig('X2D_raw.png', dpi=1200)
plt.close()

# 두 번째 그래프 생성 및 저장
plt.figure(figsize=(6, 6))
plt.scatter(X_2d1[:, 0], X_2d1[:, 1], c=y, cmap=cmap, norm=norm, s=2)
plt.xticks([])
plt.yticks([])
#plt.colorbar(label='Estimation of Action State Value', orientation='vertical')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.savefig('X2D_gnn.png', dpi=1200)
plt.close()

# 세 번째 그래프 생성 및 저장
plt.figure(figsize=(6, 6))
plt.scatter(X_2d2[:, 0], X_2d2[:, 1], c=y, cmap=cmap, norm=norm, s=2)
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.colorbar(label='Estimation of Action State Value', orientation='vertical', ticks = [])
plt.savefig('X2D_pol.png', dpi=1200)
plt.close()

# 파일 경로 반환
file_paths = {
    "raw_graph": "X2D_raw.png",
    "gnn_graph": "X2D_gnn.png",
    "pol_graph": "X2D_pol.png"
}

file_paths
