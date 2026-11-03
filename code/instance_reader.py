import networkx as nx


def read_instance(filepath: str):
    graph = nx.Graph()
    node_color = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v'):
                _, label, color = line.split()
                node_color[int(label)] = int(color)
            elif line.startswith('e'):
                _, node1, node2 = line.split()
                graph.add_edge(int(node1), int(node2))

    num_colors = len(set(node_color.values()))
    return graph, num_colors, node_color

