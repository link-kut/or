import itertools
import enum
import networkx as nx
from itertools import islice
import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from slack import WebClient
from slack.errors import SlackApiError

from main import config


client = WebClient(token=config.SLACK_API_TOKEN)


class TYPE_OF_VIRTUAL_NODE_RANKING(enum.Enum):
    TYPE_1 = 0
    TYPE_2 = 1


def get_revenue_VNR(vnr):
    revenue_cpu = sum((v_cpu_demand['CPU'] for _, v_cpu_demand in vnr.net.nodes(data=True)))

    revenue_bandwidth = sum((v_bandwidth_demand['bandwidth'] for _, _, v_bandwidth_demand in vnr.net.edges(data=True)))

    # revenue_cpu = 0.0
    # for _, v_cpu_demand in vnr.net.nodes(data=True):
    #     revenue_cpu += v_cpu_demand['CPU']
    #
    # revenue_bandwidth = 0.0
    # for _, _, v_bandwidth_demand in vnr.net.edges(data=True):
    #     revenue_bandwidth += v_bandwidth_demand['bandwidth']

    revenue = revenue_cpu + config.ALPHA * revenue_bandwidth

    return revenue


def get_cost_VNR(vnr, embedding_s_paths):
    cost_cpu = sum((v_cpu_demand['CPU'] for _, v_cpu_demand in vnr.net.nodes(data=True)))

    cost_embedded_s_path = sum(
        (len(s_links_in_path) * v_bandwidth_demand
         for _, (s_links_in_path, v_bandwidth_demand) in embedding_s_paths.items())
    )

    # cost_cpu = 0.0
    # for _, v_cpu_demand in vnr.net.nodes(data=True):
    #     cost_cpu += v_cpu_demand['CPU']

    # cost_embedded_s_path = 0.0
    # for v_link_id, (s_links_in_path, v_bandwidth_demand) in embedding_s_paths.items():
    #     cost_embedded_s_path += len(s_links_in_path) * v_bandwidth_demand

    cost = cost_cpu + config.ALPHA * cost_embedded_s_path

    return cost


def get_distance_factor_VNR(embedding_s_paths, copied_substrate):
    return 0.0


def get_attraction_strength_VNR(embedding_s_paths, copied_substrate):
    return 0.0


def get_total_hop_count_VNR(embedding_s_paths):
    total_hop_count = 0
    for _, (s_links_in_path, _) in embedding_s_paths.items():
        total_hop_count += len(s_links_in_path)
    return total_hop_count


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def step_prefix(time_step):
    return "[STEP: {0:5d}]".format(time_step)


def agent_step_prefix(agent_id, time_step):
    return "[STEP: {0:5d}/A{1}]".format(time_step, agent_id)


def run_agent_step_prefix(run, agent_id, time_step):
    return "[STEP: {0:5d}/A{1}/R{2}]".format(time_step, agent_id, run)


def send_file_to_slack(filepath):
    try:
        response = client.files_upload(
            channels='#intelligent_network',
            file=filepath
        )
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")


def peek_from_iterable(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def get_sorted_virtual_nodes_with_node_ranking(vnr, type_of_node_ranking=TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_1, beta=None):
    sorted_virtual_nodes_with_node_ranking = []

    # calculate the vnr node ranking
    for v_node_id, v_node_data in vnr.net.nodes(data=True):
        if type_of_node_ranking == TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_1:
            vnr_node_ranking = calculate_node_ranking_1(
                vnr.net.nodes[v_node_id]['CPU'],
                vnr.net[v_node_id],
                beta
            )
        elif type_of_node_ranking == TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2:
            vnr_node_ranking = calculate_node_ranking_2(
                vnr.net.nodes[v_node_id]['CPU'],
                vnr.net[v_node_id]
            )
        else:
            raise ValueError()
        sorted_virtual_nodes_with_node_ranking.append((v_node_id, v_node_data, vnr_node_ranking))

    # sorting the vnr nodes with node's ranking
    sorted_virtual_nodes_with_node_ranking.sort(
        key=lambda sorted_virtual_nodes_with_node_ranking: sorted_virtual_nodes_with_node_ranking[2], reverse=True
    )

    return sorted_virtual_nodes_with_node_ranking


def calculate_node_ranking_1(node_cpu_capacity, adjacent_links, beta):
    total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

    # total_node_bandwidth = 0.0
    # for link_id in adjacent_links:
    #     total_node_bandwidth += adjacent_links[link_id]['bandwidth']

    return beta * node_cpu_capacity + (1.0 - beta) * len(adjacent_links) * total_node_bandwidth


def calculate_node_ranking_2(node_cpu_capacity, adjacent_links):
    total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

    return node_cpu_capacity * total_node_bandwidth