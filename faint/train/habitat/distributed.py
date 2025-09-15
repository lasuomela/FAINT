from typing import Dict
from omegaconf import DictConfig

import os
import subprocess
import datetime
import torch
import numpy as np
from torch import distributed as distrib

from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size, get_ifname

# Default port to initialized the TCP store on
DEFAULT_PORT = 8738
DEFAULT_PORT_RANGE = 127
# Default address of world rank 0
DEFAULT_MASTER_ADDR = "127.0.0.1"
SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)

def get_master_addr() -> str:
    if 'MASTER_ADDR' in os.environ:
        return os.environ['MASTER_ADDR']
    elif 'SLURM_NODELIST' in os.environ:
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        return addr
    else:
        return DEFAULT_MASTER_ADDR

def init_distributed_comms():
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    local_rank, world_rank, world_size = get_distrib_size()
    
    master_port = int(os.environ.get("MASTER_PORT", DEFAULT_PORT))
    if SLURM_JOBID is not None:
        master_port += int(SLURM_JOBID) % int(
            os.environ.get("DEFAULT_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    addr = get_master_addr()

    # Torchrun will set up the TCP store for us
    # but haven't gotten it to work on slurm yet
    # so we'll set it up manually if launched with regular 
    # python launch
    is_master = world_rank == 0 if not torch.distributed.is_torchelastic_launched() else False
    tcp_store = distrib.TCPStore(  # type: ignore
        addr, master_port, world_size, is_master
    )

    distrib.init_process_group(
        'nccl',
        store=tcp_store,
        rank=world_rank,
        world_size=world_size,
        timeout=datetime.timedelta(hours=2.0) # Syncing demos to fast local storage can take a while
    )
    return local_rank, tcp_store

def init_distributed(config: DictConfig):
    """
    If running in multi gpu/node mode, set up the distributed environment
    """
    local_rank, world_rank, world_size = get_distrib_size()
    if world_size > 1:
        if not distrib.is_initialized():
            local_rank, tcp_store = init_distributed_comms()
            with read_write(config):
                config.habitat_baselines.torch_gpu_id = local_rank
                config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                config.habitat.seed += (
                    world_rank
                    * config.habitat_baselines.num_environments
                )
            # Sync here to make sure the distributed process group is properly set up.
            distrib.barrier()
    return local_rank, world_rank, world_size

def aggregate_distributed_stats(local_stats: Dict[str, np.array], world_size: int, device: torch.device):
    """Aggregates distributed stats across all workers."""

    aggregated_stats = {}
    keys = sorted(list(local_stats.keys()))
    if world_size > 1:
        distrib.barrier()
        for k in keys:
            v = local_stats[k]
            v = np.array([np.sum(v), len(v)])
            v = torch.tensor(v, dtype=torch.float32, device=device)

            s = torch.cuda.Stream()
            handle = distrib.all_reduce(v, async_op=True)
            handle.wait()
            with torch.cuda.stream(s):
                s.wait_stream(torch.cuda.default_stream())
            aggregated_stats[k] = v[0].item() / v[1].item()
        aggregated_stats['episode_count'] = v[1].item()
    else:
        for k in keys:
            aggregated_stats[k] = np.mean(local_stats[k])
        aggregated_stats['episode_count'] = len(local_stats[k])
    return aggregated_stats