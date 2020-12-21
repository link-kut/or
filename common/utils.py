import networkx as nx
from itertools import islice


def get_revenue_VNR(vnr):
    revenue_cpu = 0.0
    for node_id in vnr.nodes:
        revenue_cpu += vnr.nodes[node_id]['CPU']

    revenue_bandwidth = 0.0
    for edge_id in vnr.edges:
        revenue_bandwidth += vnr.edges[edge_id]['bandwidth']

    alpha = 0.8

    revenue = revenue_cpu + alpha * revenue_bandwidth

    return revenue


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )