import os, sys
import torch.nn.functional as F
import torch.multiprocessing as mp

from algorithms.g_a3c_gcn_vine import A3CGraphCNVNEAgent
from algorithms.model.A3C import A3C_Model
from algorithms.model.utils import SharedAdam
from main.A3C_worker import Worker

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from main.common_main import *

agents = [
    # DeterministicVNEAgent(logger),
    # RandomizedVNEAgent(logger)
    A3CGraphCNVNEAgent(0.3, logger)
    # GABaselineVNEAgent(logger)
]

agent_labels = [
    # "D-ViNE"
    # "R-ViNE"
    "A3C-GCN",
    # "GA"
]

performance_revenue = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_acceptance_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_rc_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_link_fail_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))


def main():
    gnet = A3C_Model(config.SUBSTRATE_NODES + 3, config.SUBSTRATE_NODES)  # global network Net(state_dim, action_dim)
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

if __name__ == "__main__":
    main()