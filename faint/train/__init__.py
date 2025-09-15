import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat.config.default_structured_configs import register_hydra_plugin
from omegaconf import OmegaConf

# This populates the registry with the faint configs
import faint.train.config.default_structured_configs

# Import modules that contain hydra components to add them into registry
import faint.train.habitat
import faint.train.train_utils

class TopoNavConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:

        # Get path to the direcotry of this file:
        dir_path = os.path.dirname(os.path.realpath(__file__))

        search_path.append(
            provider="toponav",
            path=f"file://{dir_path}/config/",
        )

register_hydra_plugin(TopoNavConfigPlugin)

# Enables evaluating expressions in the Hydra config yaml files
OmegaConf.register_new_resolver("eval", eval)