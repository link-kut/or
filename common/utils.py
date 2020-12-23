import networkx as nx
from itertools import islice


def get_revenue_VNR(vnr):
    revenue_cpu = 0.0

    for _, v_cpu_demand in vnr.net.nodes(data=True):
        revenue_cpu += v_cpu_demand['CPU']

    revenue_bandwidth = 0.0
    for _, _, v_bandwidth_demand in vnr.net.edges(data=True):
        revenue_bandwidth += v_bandwidth_demand['bandwidth']

    alpha = 0.8
    revenue = revenue_cpu + alpha * revenue_bandwidth

    return revenue


def get_cost_VNR(vnr, paths):
    revenue_cpu = 0.0
    for _, v_cpu_demand in vnr.net.nodes(data=True):
        revenue_cpu += v_cpu_demand['CPU']

    revenue_embedded_s_path = 0.0
    for v_link_id in paths:
        revenue_embedded_s_path += len(paths[v_link_id][0]) * paths[v_link_id][1]

    alpha = 0.8
    cost = revenue_cpu + alpha * revenue_embedded_s_path

    return cost


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )