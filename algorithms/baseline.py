import numpy as np
import networkx as nx


# Baseline Agent
class BaselineVNEAgent():
    def __init__(self):
        self.SUBSTRATE_NET = None
        self.VNR_ARRIVALS = None
        self.VNR_DICT = None

    def find_substrate_node(self, substrate, request):
        candidate_nodes = {}
        H_values = {}
        sorted_virtual_nodes = {}
        embedded_substrate_nodes = {}
        selected_nodes = []
        postponed = False

        # order the largest CPU in VNR
        for node_id, cpu_demand in request.nodes(data=True):
            sorted_virtual_nodes[node_id] = cpu_demand['CPU']
        sorted_virtual_nodes = sorted(
            sorted_virtual_nodes.items(), key=(lambda x: x[1]), reverse=True
        )

        # find the candidate node in substrate node
        for v_node_id, v_cpu_demand in sorted_virtual_nodes:
            for s_node_id, s_cpu_capacity in substrate.nodes(data=True):
                if s_cpu_capacity['CPU'] >= v_cpu_demand:
                    H_values[s_node_id] = self.calculate_node_available_resource(
                        s_cpu_capacity, substrate[s_node_id]
                    )
            if len(H_values) == 0:
                postponed = True
                break
            else:
                candidate_nodes[v_node_id] = H_values  # each candidate node’s H value
                candidate_nodes[v_node_id] = sorted(
                    candidate_nodes[v_node_id].items(), key=(lambda x: x[1]), reverse=True
                )
            H_values = {}

        # select the appropriate substrate nodes
        if not postponed:
            for v_node_id, v_cpu_demand in sorted_virtual_nodes:
                unselected_nodes = []
                for c_node_id, c_node_h in candidate_nodes[v_node_id]:
                    if c_node_id not in selected_nodes:
                        unselected_nodes.append(c_node_id)
                if len(unselected_nodes) == 0:
                    postponed = True
                else:
                    selected_node = unselected_nodes[0]
                    selected_nodes.append(selected_node)
                    embedded_substrate_nodes[v_node_id] = (selected_node, v_cpu_demand)

        return embedded_substrate_nodes, postponed

    def find_substrate_link(self, substrate, request, embedded_nodes):
        embedded_links = {}
        sorted_virtual_links = {}
        shortest_paths = {}
        embedded_nodes_id = {}
        postponed = False

        # order the largest CPU in VNR
        for src, dst, bandwidth_demand in request.edges(data=True):
            sorted_virtual_links[(src, dst)] = bandwidth_demand['bandwidth']
        sorted_virtual_links = sorted(
            sorted_virtual_links.items(), key=(lambda x: x[1]), reverse=True
        )

        # extract the embedded node id
        for v_node_id in embedded_nodes:
            for e_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                embedded_nodes_id[v_node_id] = e_node_id

        # mapping the virtual nodes and substrate nodes
        for v_link_id, bandwidth_demand in sorted_virtual_links:
            src = embedded_nodes_id[v_link_id[0]]
            dst = embedded_nodes_id[v_link_id[1]]
            shortest_paths[v_link_id] = nx.shortest_simple_paths(substrate, source=src, target=dst)

        # select the appropriate shortest path
        temp = ()
        for v_link_id, bandwidth_demand in sorted_virtual_links:
            for path in shortest_paths[v_link_id]:
                if len(path) == 2:
                    if substrate.edges[path]['bandwidth'] >= bandwidth_demand:
                        embedded_links[v_link_id] = (path, bandwidth_demand)
                        break
                elif len(path) > 2:
                    for node_id in range(len(path) - 1):
                        if substrate.edges[path[node_id], path[node_id + 1]]['bandwidth'] >= bandwidth_demand:
                            temp = (path, bandwidth_demand)
                        else:
                            postponed = True
                            break
                    embedded_links[v_link_id] = temp
                    temp = ()
                    break
            if len(embedded_links[v_link_id]) == 0:
                postponed = True
            if postponed:
                break

        return embedded_links, postponed

    # calculate the H value
    def calculate_node_available_resource(self, s_cpu_capacity, adjacent_link):
        total_node_bandwidth = 0

        for adj_link in adjacent_link:
            total_node_bandwidth += adjacent_link[adj_link]['bandwidth']

        return total_node_bandwidth + s_cpu_capacity['CPU']

    def calculate_revenue_for_a_vnr(self, request):
        total_bandwidth = 0
        total_CPU = 0

        for v_src, v_dst, v_bandwidth in request.edges(data=True):
            total_bandwidth += v_bandwidth['bandwidth']
        for v_node_id, v_cpu_demand in request.nodes(data=True):
            total_CPU += v_cpu_demand['CPU']

        return total_bandwidth + total_CPU

    def set_substrate_network(self, embedded_nodes, embedded_links):
        # set the substrate nodes
        for v_node_id in embedded_nodes:
            for e_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                self.SUBSTRATE_NET.nodes[e_node_id]['CPU'] -= cpu_demand

        # set the substrate links
        for v_link_id in embedded_links:
            for e_link_id, v_bandwidth_demand in [embedded_links[v_link_id]]:
                for node_id in range(len(e_link_id) - 1):
                    self.SUBSTRATE_NET.edges[e_link_id[node_id], e_link_id[node_id + 1]][
                        'bandwidth'] -= v_bandwidth_demand

    def get_action(self, state):
        embedded_nodes = {}
        embedded_links = {}
        action = {}

        self.SUBSTRATE_NET = state[0]

        vnr_count = 0
        for vnr in state[1:]:
            self.VNR = vnr
            embedded_nodes, postponed = self.find_substrate_node(self.SUBSTRATE_NET, self.VNR)

            if not postponed:
                embedded_links, postponed = self.find_substrate_link(self.SUBSTRATE_NET, self.VNR, embedded_nodes)

                ## Reflect the substrate network
                # print("Befor embedding")
                # print(self.SUBSTRATE_NET.nodes(data=True))
                # print(self.SUBSTRATE_NET.edges(data=True))
                # print()
                if not postponed:
                    self.set_substrate_network(embedded_nodes, embedded_links)
                    action[vnr_count] = []
                    action[vnr_count].append({
                        "embedded_nodes": embedded_nodes,
                        "embedded_links": embedded_links,
                        "postponed": postponed
                    })
                # print("Aefor embedding")
                # print(self.SUBSTRATE_NET.nodes(data=True))
                # print(self.SUBSTRATE_NET.edges(data=True))
                # print()
            if postponed:
                action[vnr_count] = []
                action[vnr_count].append({
                    "embedded_nodes": {},
                    "embedded_links": {},
                    "postponed": postponed
                })
            vnr_count += 1

        return action