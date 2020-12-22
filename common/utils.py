import networkx as nx
from itertools import islice


def get_revenue_VNR(vnr):
    revenue_cpu = 0.0

    for _, v_cpu_demand in vnr["graph"].nodes(data=True):
        revenue_cpu += v_cpu_demand['CPU']

    revenue_bandwidth = 0
    for _, _, v_bandwidth_demand in vnr["graph"].edges(data=True):
        revenue_bandwidth += v_bandwidth_demand['bandwidth']

    alpha = 0.8
    revenue = revenue_cpu + alpha * revenue_bandwidth

    return revenue


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )