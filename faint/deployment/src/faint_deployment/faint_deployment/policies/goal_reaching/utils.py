from typing import Tuple, Dict, Union

import numpy as np
import torch
import huggingface_hub

def load_model(
        model_config: Dict,
        device,
        logger,
    ) -> torch.jit.ScriptModule:
    """
    Load a Torchscript model from a checkpoint file
    """

    if not model_config['checkpoint_path'].exists():
        if not 'huggingface_repo_id' in model_config:
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_config['checkpoint_path']}",
            )
        logger.info(
            f"Downloading model checkpoint {model_config['checkpoint_path'].name} from Huggingface Hub"
        )
        huggingface_hub.snapshot_download(
            model_config['huggingface_repo_id'],
            local_dir=model_config['checkpoint_path'].parent,
            repo_type='model',
        )
        if not model_config['checkpoint_path'].exists():
            raise FileNotFoundError(
                f"""Model checkpoint {model_config['checkpoint_path'].name}
                not found in Huggingface Hub repo {model_config['huggingface_repo_id']}
                """)

    model = torch.jit.optimize_for_inference(
        torch.jit.load(
            model_config['checkpoint_path'],
            map_location=device,
        ),
    )
    model.eval()
    return model

def clip_angle(self, theta) -> float:
    """
    Clip angle to [-pi, pi]
    """
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi

def pd_control(
    waypoint: np.ndarray,
    DT: float,
    MAX_V: float,
    MAX_W: float,
    EPS: float = 1e-8,
    ) -> Tuple[float, float]:
    """
    Convert waypoint to robot command velocities using a PD controller
    """
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint

    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / DT
    elif np.abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = dx / DT
        w = np.arctan(dy / dx) / DT

    v = np.clip(v, 0, MAX_V)
    w = np.clip(w, -MAX_W, MAX_W)

    return v, w