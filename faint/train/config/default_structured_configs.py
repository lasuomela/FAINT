"""
The default configs for the topological navigation imitation learning pipeline
"""
from typing import List, Dict, Any

from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import II

from habitat.config.default_structured_configs import (
    LabSensorConfig,
    ActionConfig,
    TopDownMapMeasurementConfig,
    DatasetConfig,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfig,
    HabitatBaselinesBaseConfig,
    EvalConfig,
)

"""
Habitat sensor and action configs
"""
@dataclass
class TopDownTopoMapMeasurementConfig(TopDownMapMeasurementConfig):
    type: str = "TopDownTopoMap"

@dataclass
class SubgoalSensorConfig(LabSensorConfig):
    """
    Sensor that samples subgoals along the path to an episode goal.
    """
    type: str = "SubgoalSensor"
    planner_safety_margin: float = 0.3 # meters, additional margin to obstacles for path planning
    can_fail: bool = True # Flag indicating that failure in sensor to find a valid path in a episode should trigger environment reset
    align_agent: bool = True # If True, the agent will be aligned with the path direction at episode start
    controller_lookahead: float = 0.1 # meters, lookahead distance for the oracle controller
    controller_angular_error_threshold: float = np.pi/4 # With orientation difference greater than this threshold, the controller will rotate in place
    smooth_acceleration: bool = True # Make the agent accelerate smoothly

    # Subgoal sampling parameters
    subgoal_sampling_strategy: str = II("toponav.augmentation.subgoal_sampling_strategy") # "uniform" | "random" | "recast_corners"
    subgoal_spacing: float = II("toponav.augmentation.subgoal_spacing") # meters, for use with "uniform" strategy
    subgoal_min_spacing: float = II("toponav.augmentation.subgoal_min_spacing") # meters, for use with "random" strategy
    subgoal_max_spacing: float = II("toponav.augmentation.subgoal_max_spacing") # meters, for use with "random" strategy

    # These are provided by the final parsed config
    max_linear_speed: float = II("habitat_baselines.il.expert_agent.max_linear_vel") # m/s, maximum linear speed of the agent
    max_turn_speed: float = II("habitat_baselines.il.expert_agent.max_angular_vel") # rad/s, maximum turning speed of the agent
    time_step: float = II("habitat.task.actions.se2_velocity_action.time_step") # seconds, simulation step duration
    action_pred_horizon: int = II("habitat_baselines.il.trainer.action_pred_horizon") # Number of steps for which to roll out the oracle at each time step

@dataclass
class ImageSubgoalSensorConfig(SubgoalSensorConfig):
    """
    Generates image observations of the sampled subgoals.
    """
    type: str = "ImageSubgoalSensor"
    obs_type: str = II('toponav.image_sensor.obs_type') # "rgb" | "depth"

@dataclass
class SE2VelocityActionConfig(ActionConfig):
    """
    Action space for SE2 continuous velocity control.
    """
    type: str = "SE2VelocityAction"
    lin_vel_range: List[float] = field(default_factory=lambda: [-0.5, 0.25]) # meters/sec
    ang_vel_range: List[float] = field(default_factory=lambda: [-10.0, 10.0]) # deg/sec
    time_step: float = 1.0  # seconds, simulation step duration
    timestep_noise_multiplier: float = II("toponav.augmentation.timestep_noise_multiplier") # Noise factor by which to vary the simulation time step length

"""
Habitat baselines configs
"""
@dataclass
class HabitatBaselinesBaseConfig:
    def __post_init__(self):
        # Type checking for each attribute using type hints
        for field_name, field_type in self.__annotations__.items():
            if field_name != "self":  # Exclude 'self' from type checking
                assigned_value = getattr(self, field_name)
                try:
                    assert isinstance(assigned_value, field_type) or (isinstance(assigned_value, str) and (assigned_value[0] == '$')), f"{field_name} must be of type {field_type.__name__}, is {getattr(self, field_name)}"
                except TypeError as e:
                    pass
@dataclass
class AugmentationConfig(HabitatBaselinesBaseConfig):
    """
    Config to aggregate all augmentation parameters for the simulation
    """
    type: str = "None"

    test_time_augment: bool = False

    # Subgoal sampling parameters
    subgoal_sampling_strategy: str = "random" # "uniform" | "random" | "recast_corners"
    subgoal_spacing: float = 1.5 # meters, for use with "uniform" strategy
    subgoal_min_spacing: float = 0.5 # meters, for use with "random" strategy
    subgoal_max_spacing: float = 3.0 # meters, for use with "random" strategy

    # SE2VelocityAction noise
    timestep_noise_multiplier: float = 0.0
    
@dataclass
class ImageSensorConfig(HabitatBaselinesBaseConfig):
    """
    Config to aggregate all relevant image sensor related parameters
    """
    height: int = 224
    width: int = 224
    hfov: int = 110
    position: List[float] = field(default_factory=lambda: [0.0, 0.38, 0.0])
    obs_type: str = 'rgb' # 'rgb' | 'depth'
    normalize_depth: bool = False
    min_depth: float = 0.0
    max_depth: float = 10.0

"""
Agent configs
"""

@dataclass
class AgentBaseConfig(HabitatBaselinesBaseConfig):
    # Agent velocity limits
    max_linear_vel: float = 0.31 # m/s
    max_angular_vel: float = 1.9 # unsigned magnitude rad/s

    # For policies that utilize output parametrization as waypoints
    target_waypoint_idx: int = 2

    # Thresholds determining if the agent has reached the episode goal
    success_distance: float = 0.2 # meters
    success_rotation: float = np.pi/4 # radians


@dataclass
class OracleVelocityAgentConfig(AgentBaseConfig):
    type: str = "OracleVelocityAgent"

@dataclass
class ImitationAgentConfig(AgentBaseConfig):
    type: str = ""
    goal_sensor_key: str = "imagesubgoal"
    obs_key: str = II('toponav.image_sensor.obs_type') # "rgb" | "depth"
    time_step: float = II('habitat.task.actions.se2_velocity_action.time_step') # Simulation step duration

"""
Policy configs
"""
@dataclass
class PolicyBaseConfig(HabitatBaselinesBaseConfig):
    type: str = ""

@dataclass
class OracleVelocityPolicyConfig(PolicyBaseConfig):
    type: str = "OracleVelocityPolicy"
    input_height: int = II("toponav.image_sensor.height")
    input_width: int = II("toponav.image_sensor.width")

@dataclass
class ImitationPolicyConfig(PolicyBaseConfig):
    """
    Base config for imitation learning policies
    """
    input_height: int = II("toponav.image_sensor.height")
    input_width: int = II("toponav.image_sensor.width")

    checkpoint: Path = Path() # This evaluates to cwd

    # Encoder
    pretrained_encoder: bool = True
    freeze_encoder: bool = False

    # Output parametrization
    output_type: str = 'waypoints' # 'continuous' or 'waypoints'
    action_pred_horizon: int = II("habitat_baselines.il.trainer.action_pred_horizon")

    # Learning rate
    base_lr: float = 1e-3
    loss_mode: str = "last" # If the loss should be computed over the last timestep or the full sequence

    # By default, apply the learning rate scheduler in 'lr_sceduler' from the first epoch
    no_scheduler_epochs: int = 0

    # Scheduler for epochs after no_scheduler_epochs
    lr_scheduler: Dict[str, Any] = field(default_factory=lambda: dict(
        _target_="torch.optim.lr_scheduler.CosineAnnealingLR",
        T_max=3, 
        eta_min=1e-6,
        )
    )

@dataclass
class ViNTPolicyConfig(ImitationPolicyConfig):
    type: str = "ViNTModel"
    sequence_length: int = II("habitat_baselines.il.trainer.sequence_length")
    output_type: str = 'waypoints'
    encoder: str = "efficientnet_b0"
    encoding_dim: int = 512
    input_height: int = 126
    input_width: int = 224

@dataclass
class FAINTPolicyConfig(ImitationPolicyConfig):
    type: str = "FAINTModel"
    sequence_length: int = II("habitat_baselines.il.trainer.sequence_length")
    output_type: str = 'waypoints'
    input_height: int = 224
    input_width: int = 224

    encoder: str = "theaiinstitute/theia-tiny-patch16-224-cddsv"
    freeze_encoder: bool = True

    # How to flatten the encoder output
    compression_channels: int = 2
    compression_type: str = "flatten" # "flatten" | "mean" <- meanpool

    # Binocular encoder that conditions goal image with latest observation
    obsgoal_fusion_type: str = "CrossBlock"
    obsgoal_fusion_num_attn_heads: int = 4
    obsgoal_fusion_num_attn_layers: int = 4
    obsgoal_fusion_ff_dim_factor: int = 2

    # Transformer that processes the sequence of observations and goal.
    seq_encoder_num_attn_heads: int = 4
    seq_encoder_num_attn_layers: int = 4
    seq_encoder_ff_dim_factor: int = 2
    seq_encoding_type: str = "cls"
    seq_use_cls_token: bool = True
    seq_pos_enc_type: str = "learned"

    # MLP predictor head
    prediction_head_layer_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    prediction_head_dropout: float = 0.2

"""
Imitation learning configs
"""

@dataclass
class ILTrainerConfig(HabitatBaselinesBaseConfig):
    sequence_length: int = 1
    sequence_stride: int = 1
    action_pred_horizon: int = 1 # Number of actions to predict at each time step
    batch_size: int = 64
    num_workers: int = 6 # Dataloader workers
    num_epochs_per_round: int = 1 # Epochs per dagger round
    accelerator: str = "gpu"
    num_devices: int = 1 # num gpus per node
    num_nodes: int = 1
    save_top_k: int = 3 # Number of top checkpoints to save as measured by val loss
    check_val_every_n_epoch: int = 1
    val_check_interval: float = 1.0
    closed_loop_eval_every_round: bool = True
    skip_train_until: int = 0 # DAgger rounds to skip training. For debugging
    gradient_clip_val: float = 0.0
    gpu_augmentation: bool = False # If image augmentations should be applied on GPU

@dataclass
class ILDataCollectionConfig(HabitatBaselinesBaseConfig):
    scratch_dir: Path = Path("dagger_examples") # Path to save demonstrations
    fast_tmp_dir: str = "" # A temporary fast disk to store trajectories during training
    recollect_train_demos: bool = False # Overwrite existing demonstrations
    recollect_val_demos: bool = False # Overwrite existing demonstrations
    num_steps_per_round: int = -1 # Number of steps to collect per round
    num_episodes_per_round: int = -1 # Number of episodes to collect per round
    num_episode_repeats: int = 1 # Number of times to repeat each episode
    trajectory_min_length: int = 10 # Minimum length of a trajectory to save
    save_demonstrations: bool = True
    num_workers: int = 2 # For TrajectoryAccumulator disk writer 
    chunk_size: str = "1GB" # Dataset chunk size to write on disk
    success_distance: float = II("habitat.task.measurements.success.success_distance")

@dataclass
class ILConfig(HabitatBaselinesBaseConfig):
    """
    Base experiment config for imitation learning
    """
    type: str = 'bc'
    num_rounds: int = 1
    dagger_beta_decay: float = 0.8
    trainer: ILTrainerConfig = ILTrainerConfig()
    data_collection: ILDataCollectionConfig = ILDataCollectionConfig()

    expert_agent: AgentBaseConfig = AgentBaseConfig()
    student_agent: AgentBaseConfig = AgentBaseConfig()
    eval_agent: AgentBaseConfig = AgentBaseConfig()

    expert_policy: PolicyBaseConfig = PolicyBaseConfig()
    student_policy: PolicyBaseConfig = PolicyBaseConfig()
    eval_policy: PolicyBaseConfig = PolicyBaseConfig()

    log_metrics: bool = False
    eval_save_results: bool = False

@dataclass
class EvalConfig(EvalConfig):
    eval_type: str = "online"

@dataclass
class HabitatBaselinesConfig(HabitatBaselinesConfig):
    """
    Redefine the HabitatBaselinesConfig to include the imitation learning config
    """
    trainer_name: str = "dagger"
    eval: EvalConfig = EvalConfig()
    checkpoint_folder: Path = Path()

@dataclass
class TopoNavILConfig(HabitatBaselinesConfig):
    il: ILConfig = ILConfig()

@dataclass
class TopoNavDatasetConfig(DatasetConfig):
    # If the scence episode list should be subsampled
    episodes_per_scene: int = -1

"""
Store the configs in the ConfigStore
"""
cs = ConfigStore.instance()
cs.store(
    package="habitat.task.actions.se2_velocity_action",
    group="habitat/task/actions",
    name="se2_velocity_action",
    node=SE2VelocityActionConfig
)
cs.store(
    package="habitat.task.lab_sensors.subgoal_tracker",
    group="habitat/task/lab_sensors",
    name="subgoal_tracker",
    node=SubgoalSensorConfig
)
cs.store(
    package="habitat.task.lab_sensors.imagesubgoal",
    group="habitat/task/lab_sensors",
    name="imagesubgoal",
    node=ImageSubgoalSensorConfig
)
cs.store(
    package="habitat.task.measurements.top_down_topomap",
    group="habitat/task/measurements",
    name="top_down_topomap",
    node=TopDownTopoMapMeasurementConfig
)
cs.store(
    group="habitat_baselines",
    name="toponav_il_config_base",
    node=TopoNavILConfig,
)
cs.store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="dataset_config_schema",
    node=TopoNavDatasetConfig,
)
cs.store(
    group='toponav',
    name="oracle_velocity_policy",
    node=OracleVelocityPolicyConfig,
)
cs.store(group='toponav',
         name="imitation_vint_policy",
         node=ViNTPolicyConfig,
)
cs.store(group='toponav',
         name="imitation_faint_policy",
         node=FAINTPolicyConfig,
)
cs.store(group='toponav',
         name="imitation_agent",
         node=ImitationAgentConfig,
)
cs.store(group='toponav',
         name="oracle_velocity_agent",
         node=OracleVelocityAgentConfig,
)
cs.store(package='toponav.augmentation',
        group='toponav',
         name="augmentation",
         node=AugmentationConfig,
)
cs.store(package='toponav.image_sensor',
        group='toponav',
         name="image_sensor",
         node=ImageSensorConfig,
)

from habitat.config.default_structured_configs import HabitatSimFisheyeRGBSensorConfig
cs.store(
    group="habitat/simulator/sim_sensors",
    name="fisheye_rgb_sensor",
    node=HabitatSimFisheyeRGBSensorConfig,
)

