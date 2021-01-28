from algorithms.baseline import BaselineVNEAgent
from common import utils
from main import config
import copy
import networkx as nx
import pulp as plp
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

class DeterministicVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(DeterministicVNEAgent, self).__init__(logger)

    def find_subset_S_for_virtual_node(self, copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes):
        '''
        find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
        :param substrate: substrate network
        :param v_cpu_demand: cpu demand of the given virtual node
        :return:
        '''

        subset_S = (s_node_id for s_node_id, s_node_data in copied_substrate.net.nodes(data=True)
                    if s_node_data['CPU'] >= v_cpu_demand and s_node_id not in already_embedding_s_nodes and s_node_data['LOCATION'] == v_node_location)

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

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id:
                    sum(opt_lp_f_vars[opt_lp_f_vars['uv'] == (s_node_id, v_node_id + config.SUBSTRATE_NODES)]['solution_value']) *
                    opt_lp_x_vars[(opt_lp_x_vars['w'] == s_node_id) &
                                  (opt_lp_x_vars['m'] == v_node_id + config.SUBSTRATE_NODES)]['solution_value'].values,
                default=None
            )

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

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        # total_node_bandwidth = 0.0
        # for link_id in adjacent_links:
        #     total_node_bandwidth += adjacent_links[link_id]['bandwidth']

        return 0.3 * node_cpu_capacity + (1.0 - 0.3) * len(adjacent_links) * total_node_bandwidth

    def calculate_LP_variables(self, augmented_substrate, vnr):
        set_i = range(len(list(vnr.net.edges)))
        set_uv = list(augmented_substrate.net.edges)
        set_w = []
        set_m = []

        a_remain_bandwidth = {}
        v_remain_bandwidth = {}
        remain_CPU_w = {}
        remain_CPU_m = {}

        for a_edge_src, a_edge_dst, a_edge_data in augmented_substrate.net.edges(data=True):
            a_remain_bandwidth[(a_edge_src, a_edge_dst)] = a_edge_data['bandwidth']

        count = 0
        min_bandwidth = 100
        for v_edge_src, v_edge_dst, v_edge_data in vnr.net.edges(data=True):
            v_remain_bandwidth[count] = v_edge_data['bandwidth']
            if min_bandwidth >= v_edge_data['bandwidth']:
                min_bandwidth = v_edge_data['bandwidth']
            count += 1

        # f_vars
        f_vars = {
            (i,uv): plp.LpVariable(
                cat=plp.LpContinuous,
                lowBound=0,
                name="f_{0}_{1}".format(i, uv)
            )
            for i in set_i for uv in set_uv
        }

        for a_node_id, a_node_data in augmented_substrate.net.nodes(data=True):
            if a_node_id < config.SUBSTRATE_NODES:
                set_w.append(a_node_id)
                remain_CPU_w[a_node_id] = a_node_data['CPU']
            else:
                set_m.append(a_node_id)
                remain_CPU_m[a_node_id] = a_node_data['CPU']

        # x_vars
        x_vars = {(w,m):
                plp.LpVariable(
                    cat=plp.LpContinuous,
                    lowBound=0, upBound=1,
                    name="x_{0}_{1}".format(w, m)
                )
                for w in set_w for m in set_m
        }

        opt_model = plp.LpProblem(name="MIP Model")

        # Constraints 1
        constraints = {}
        for uv in set_uv:
            constraints[uv] = opt_model.addConstraint(
                    plp.LpConstraint(
                        e=plp.lpSum(f_vars[i,uv] for i in set_i),
                        sense=plp.LpConstraintGE,
                        rhs=min_bandwidth,
                        name="constraint_1_{0}".format(uv)
                    )
                )

        # Constraints 2
        for w in set_w:
            constraints[w] = opt_model.addConstraint(
                plp.LpConstraint(
                    e=plp.lpSum(remain_CPU_m[m] * x_vars[w, m] for m in set_m),
                    sense=plp.LpConstraintLE,
                    rhs=remain_CPU_w[w],
                    name="constraint_2_{0}".format(w)
                )
            )

        # Objective function
        objective = plp.lpSum(
            f_vars[i, uv] * 1 / (a_remain_bandwidth[uv] + 0.000001) for i in set_i for uv in set_uv
        )
        objective += plp.lpSum(
            x_vars[(w, m)] * remain_CPU_m[m] * 1 / (remain_CPU_w[w] + 0.000001) for w in set_w for m in set_m
        )
        # objective = plp.lpSum(
        #     f_vars[i, uv] for i in set_i for uv in set_uv
        # )
        # objective += plp.lpSum(
        #     x_vars[(w, m)] * remain_CPU_m[m] for w in set_w for m in set_m
        # )

        # for minimization
        # solve VNE_LP_RELAX
        opt_model.sense = plp.LpMinimize
        opt_model.setObjective(objective)
        opt_model.solve(plp.GLPK_CMD(msg=0))

        # make the DataFrame for f_vars and x_vars
        opt_lp_f_vars = pd.DataFrame.from_dict(f_vars, orient="index", columns=['variable_object'])
        opt_lp_f_vars.index = pd.MultiIndex.from_tuples(opt_lp_f_vars.index, names=["i", "uv"])
        opt_lp_f_vars.reset_index(inplace=True)
        opt_lp_f_vars["solution_value"] = opt_lp_f_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_f_vars.drop(columns=["variable_object"], inplace=True)

        opt_lp_x_vars = pd.DataFrame.from_dict(x_vars, orient="index", columns=['variable_object'])
        opt_lp_x_vars.index = pd.MultiIndex.from_tuples(opt_lp_x_vars.index, names=["w", "m"])
        opt_lp_x_vars.reset_index(inplace=True)
        opt_lp_x_vars["solution_value"] = opt_lp_x_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_x_vars.drop(columns=["variable_object"], inplace=True)

        return opt_lp_f_vars, opt_lp_x_vars
