from algorithms.baseline import BaselineVNEAgent
from common import utils
from main import config
import copy
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import pulp as plp
import pandas as pd
import numpy as np
import sys

import warnings
warnings.filterwarnings(action='ignore')

class RandomizedVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(RandomizedVNEAgent, self).__init__(logger)

    def find_subset_S_for_virtual_node(self, copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes):
        '''
        find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
        :param substrate: substrate network
        :param v_cpu_demand: cpu demand of the given virtual node
        :return:
        '''

        subset_S = (s_node_id for s_node_id, s_node_data in copied_substrate.net.nodes(data=True)
                    if s_node_data['CPU'] >= v_cpu_demand and s_node_id not in already_embedding_s_nodes and s_node_data['LOCATION'] == v_node_location)

        # subset_S = []
        # for s_node_id, s_cpu_capacity in copied_substrate.net.nodes(data=True):
        #     if s_cpu_capacity['CPU'] >= v_cpu_demand and \
        #             s_node_id not in already_embedding_s_nodes and \
        #             s_cpu_capacity['LOCATION'] == v_node_location:
        #         subset_S.append(s_node_id)

        return subset_S

    def find_substrate_nodes(self, copied_substrate, vnr):
        '''
        Execute Step 1
        :param copied_substrate: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodes
        '''
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        # Generate the augmented substrate network with location info.
        augmented_substrate = copy.deepcopy(copied_substrate)
        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']
            # Meta node add
            augmented_substrate.net.add_node(v_node_id + config.SUBSTRATE_NODES)
            augmented_substrate.net.nodes[v_node_id + config.SUBSTRATE_NODES]['CPU'] = v_cpu_demand
            augmented_substrate.net.nodes[v_node_id + config.SUBSTRATE_NODES]['LOCATION'] = v_node_location
            # Meta edge add
            for a_node_id, a_node_data, in augmented_substrate.net.nodes(data=True):
                a_cpu_demand = a_node_data['CPU']
                a_node_location = a_node_data['LOCATION']
                if v_node_location == a_node_location and a_node_id < config.SUBSTRATE_NODES:
                    augmented_substrate.net.add_edge(v_node_id + config.SUBSTRATE_NODES, a_node_id)
                    augmented_substrate.net.edges[v_node_id + config.SUBSTRATE_NODES, a_node_id].update({'bandwidth': 1000000})

        opt_lp_f_vars, opt_lp_x_vars = self.calculate_LP_variables(augmented_substrate, vnr)

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # selected_s_node_id = max(
            #     subset_S_per_v_node[v_node_id],
            #     key=lambda s_node_id:
            #         sum(opt_lp_f_vars[(opt_lp_f_vars['u'] == s_node_id) &
            #                           (opt_lp_f_vars['v'] == v_node_id + config.SUBSTRATE_NODES)]['solution_value'].values +
            #             opt_lp_f_vars[(opt_lp_f_vars['u'] == v_node_id + config.SUBSTRATE_NODES) &
            #                           (opt_lp_f_vars['v'] == s_node_id)]['solution_value'].values
            #             ) *
            #         opt_lp_x_vars[(opt_lp_x_vars['u'] == s_node_id) &
            #                       (opt_lp_x_vars['v'] == v_node_id + config.SUBSTRATE_NODES)]['solution_value'].values,
            #     default=None
            # )

            # for calculating p_value
            selected_s_node_p_value = []
            candidate_s_node_id = []
            for s_node_id in subset_S_per_v_node[v_node_id]:
                candidate_s_node_id.append(s_node_id)
                selected_s_node_p_value.append(
                    sum(opt_lp_f_vars[(opt_lp_f_vars['u'] == s_node_id) &
                                  (opt_lp_f_vars['v'] == v_node_id + config.SUBSTRATE_NODES)]['solution_value'].values +
                    opt_lp_f_vars[(opt_lp_f_vars['u'] == v_node_id + config.SUBSTRATE_NODES) &
                                  (opt_lp_f_vars['v'] == s_node_id)]['solution_value'].values))

            # Calculate the probability
            total_p_value = sum(selected_s_node_p_value)
            if total_p_value == 0:
                self.num_node_embedding_fails += 1
                msg = "VNR REJECTED ({0}): 'no suitable NODE for CPU demand: {1}' {2}".format(
                    self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None
            else:
                probability = selected_s_node_p_value / total_p_value
                selected_s_node_id = np.random.choice(candidate_s_node_id, p=probability)

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR REJECTED ({0}): 'no suitable NODE for CPU demand: {1}' {2}".format(
                    self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        embedding_s_paths = {}
        temp_copied_substrate = copied_substrate.net.to_directed()

        # mapping the virtual nodes and substrate_net nodes
        for src_v_node, dst_v_node, edge_data in vnr.net.edges(data=True):
            v_link = (src_v_node, dst_v_node)
            src_s_node = embedding_s_nodes[src_v_node][0]
            dst_s_node = embedding_s_nodes[dst_v_node][0]
            v_bandwidth_demand = edge_data['bandwidth']

            if src_s_node == dst_s_node:
                s_links_in_path = []
                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)
            else:
                subnet = nx.subgraph_view(
                    copied_substrate.net,
                    filter_edge=lambda node_1_id, node_2_id: \
                        True if copied_substrate.net.edges[(node_1_id, node_2_id)][
                                    'bandwidth'] >= v_bandwidth_demand else False
                )

                # Just for assertion
                # for u, v, a in subnet.edges(data=True):
                #     assert a["bandwidth"] >= v_bandwidth_demand

                if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                    self.num_link_embedding_fails += 1
                    msg = "VNR REJECTED ({0}): 'no suitable LINK for bandwidth demand: {1}' {2}".format(
                        self.num_link_embedding_fails, v_bandwidth_demand, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                MAX_K = 1

                # shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]

                residual_network = shortest_augmenting_path(temp_copied_substrate, src_s_node, dst_s_node,
                                                            capacity='bandwidth',
                                                            cutoff=v_bandwidth_demand)
                s_links_in_path = []
                path = []
                for src_r_node, dst_r_node, r_edge_data in residual_network.edges(data=True):
                    if r_edge_data['flow'] > 0:
                        s_links_in_path.append((src_r_node, dst_r_node))

                # s_links_in_path = []
                # for node_idx in range(len(shortest_s_path) - 1):
                #     s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                for s_link in s_links_in_path:
                    # assert copied_substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                    copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)

        return embedding_s_paths

    def calculate_LP_variables(self, augmented_substrate, vnr):
        num_nodes = len(list(augmented_substrate.net.nodes))
        edges_bandwidth = [[0] * num_nodes for _ in range(num_nodes)]
        a_nodes_id = []
        s_nodes_id = []
        meta_nodes_id = []
        nodes_CPU = []
        v_flow_id = []
        v_flow_start = []
        v_flow_end = []
        v_flow_demand = []

        for a_edge_src, a_edge_dst, a_edge_data in augmented_substrate.net.edges(data=True):
            edges_bandwidth[a_edge_src][a_edge_dst] = a_edge_data['bandwidth']
            edges_bandwidth[a_edge_dst][a_edge_src] = a_edge_data['bandwidth']

        for a_node_id, a_node_data in augmented_substrate.net.nodes(data=True):
            a_nodes_id.append(a_node_id)
            nodes_CPU.append(a_node_data['CPU'])
            if a_node_id >= config.SUBSTRATE_NODES:
                meta_nodes_id.append(a_node_id)
            else:
                s_nodes_id.append(a_node_id)

        id_idx = 0
        for v_edge_src, v_edge_dst, v_edge_data in vnr.net.edges(data=True):
            v_flow_id.append(id_idx)
            v_flow_start.append(v_edge_src + config.SUBSTRATE_NODES)
            v_flow_end.append(v_edge_dst + config.SUBSTRATE_NODES)
            v_flow_demand.append(v_edge_data['bandwidth'])
            id_idx += 1

        # f_vars
        f_vars = {
            (i,u,v): plp.LpVariable(
                cat=plp.LpContinuous,
                lowBound=0,
                name="f_{0}_{1}_{2}".format(i, u, v)
            )
            for i in v_flow_id for u in a_nodes_id for v in a_nodes_id
        }

        # x_vars
        x_vars = {(u,v):
                plp.LpVariable(
                    cat=plp.LpContinuous,
                    lowBound=0, upBound=1,
                    name="x_{0}_{1}".format(u, v)
                )
                for u in a_nodes_id for v in a_nodes_id
        }

        opt_model = plp.LpProblem(name="MIP Model", sense=plp.LpMinimize)


        # Objective function
        opt_model += sum(edges_bandwidth[u][v] / (edges_bandwidth[u][v] + 0.000001) *
                              sum(f_vars[i,u,v] for i in v_flow_id)
                              for u in s_nodes_id for v in s_nodes_id) + \
                     sum(nodes_CPU[w] / (nodes_CPU[w] + 0.000001) *
                              sum(x_vars[m, w] * nodes_CPU[m]
                              for m in meta_nodes_id) for w in s_nodes_id)

        # Capacity constraint 1
        for u in a_nodes_id:
            for v in a_nodes_id:
                opt_model += sum(f_vars[i,u,v] + f_vars[i,v,u] for i in v_flow_id) <= edges_bandwidth[u][v]

        # Capacity constraint 2
        for m in meta_nodes_id:
            for w in s_nodes_id:
                opt_model += nodes_CPU[w] >= x_vars[m, w] * nodes_CPU[m]

        # Flow constraints 1
        for i in v_flow_id:
            for u in s_nodes_id:
                opt_model += sum(f_vars[i,u,w] for w in a_nodes_id) - \
                             sum(f_vars[i,w,u] for w in a_nodes_id) == 0

        # Flow constraints 2
        for i in v_flow_id:
            for fs in v_flow_start:
                opt_model += sum(f_vars[i,fs,w] for w in a_nodes_id) - \
                             sum(f_vars[i,w,fs] for w in a_nodes_id) == v_flow_demand[i]

        # Flow constraints 3
        for i in v_flow_id:
            for fe in v_flow_end:
                opt_model += sum(f_vars[i,fe,w] for w in a_nodes_id) - \
                             sum(f_vars[i,w,fe] for w in a_nodes_id) == -1 * v_flow_demand[i]

        # Meta constraint 1
        for w in s_nodes_id:
            opt_model += sum(x_vars[m,w] for m in meta_nodes_id) <= 1

        # Meta constraint 2
        for u in a_nodes_id:
            for v in a_nodes_id:
                opt_model += x_vars[u,v] == x_vars[v,u]

        # for minimization
        # solve VNE_LP_RELAX
        # opt_model.solve(plp.GLPK_CMD(msg=1))
        opt_model.solve(plp.PULP_CBC_CMD(msg=0))

        # for v in opt_model.variables():
        #     if v.varValue > 0:
        #         print(v.name, "=", v.varValue)

        # make the DataFrame for f_vars and x_vars
        opt_lp_f_vars = pd.DataFrame.from_dict(f_vars, orient="index", columns=['variable_object'])
        opt_lp_f_vars.index = pd.MultiIndex.from_tuples(opt_lp_f_vars.index, names=["i", "u", "v"])
        opt_lp_f_vars.reset_index(inplace=True)
        opt_lp_f_vars["solution_value"] = opt_lp_f_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_f_vars.drop(columns=["variable_object"], inplace=True)

        opt_lp_x_vars = pd.DataFrame.from_dict(x_vars, orient="index", columns=['variable_object'])
        opt_lp_x_vars.index = pd.MultiIndex.from_tuples(opt_lp_x_vars.index, names=["u", "v"])
        opt_lp_x_vars.reset_index(inplace=True)
        opt_lp_x_vars["solution_value"] = opt_lp_x_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_x_vars.drop(columns=["variable_object"], inplace=True)

        return opt_lp_f_vars, opt_lp_x_vars
