import matplotlib.pyplot as plt
import networkx as nx

# 노드와 엣지 데이터
# nodes = [1, 3, 4, 5, 6, 10, 2, 3, 1, 4]
# edges = [[0, 3, 2, 2, 5, 7, 2, 3], [1, 2, 4, 3, 4, 2, 3, 1]]
# edges2 = [[5, 7, 2, 3, 1, 3, 2, 2], [1, 2, 4, 3, 4, 2, 3, 1]]

def visualize_heterogeneous_graph(nodes, edges, edges2):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for i in range(len(edges[0])):
        G.add_edge(edges[0][i], edges[1][i])
    #print(nodes)
    # 노드에 따른 색상 지정 함수 (RGB 코드 반환)
    def generate_rgb_code(value):
        # 범위를 0과 16 사이로 제한
        value = max(0, min(value, 16))

        # 정규화된 값을 계산
        normalized_value = value / 16.0

        # RGB 코드 계산
        red = int(255 * (1 - normalized_value))/255
        green = int(255 * (1 - normalized_value))/255
        blue = int(255 * normalized_value)/255

        # RGB 코드를 문자열로 반환
        return (red, green, blue)
    # 그래프 레이아웃 설정
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

    # 첫 번째 엣지 리스트를 실선으로 그림
    nx.draw(G, pos, edgelist=list(zip(edges[0], edges[1])), with_labels=True,
            node_size=500, node_color=[generate_rgb_code(node) for node in G.nodes()], font_size=10, font_color='black', edge_color='black', style='solid')

    # 두 번째 엣지 리스트를 점선으로 그림
    nx.draw(G, pos, edgelist=list(zip(edges2[0], edges2[1])), with_labels=False,
            node_size=500, node_color=[generate_rgb_code(node) for node in G.nodes()], font_size=10, font_color='black', edge_color='red', style='dashed')

    plt.title("그래프")
    plt.show()

if __name__ == "__main__":
    pass