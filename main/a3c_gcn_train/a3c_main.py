import os, sys
import torch.multiprocessing as mp

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from main.a3c_gcn_train.a3c_worker import Worker
from algorithms.model.A3C import A3C_Model
from algorithms.model.utils import SharedAdam
from common import config


def main():
    global_net = A3C_Model(
        chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES, action_dim=config.SUBSTRATE_NODES
    )
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    mp.set_start_method('spawn')

    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.0)
    message_queue = mp.Queue()

    # parallel training
    # print("Number of Workers: ", mp.cpu_count())
    # workers = [Worker(global_net, optimizer, global_episode, global_episode_reward, message_queue, i) for i in range(mp.cpu_count())]
    workers = [
        Worker(
            global_net, optimizer, global_episode, global_episode_reward, message_queue, idx, PROJECT_HOME
        ) for idx in range(config.NUM_WORKERS)
    ]

    for w in workers:
        w.start()

    res = []  # record episode reward to plot
    while True:
        r = message_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
