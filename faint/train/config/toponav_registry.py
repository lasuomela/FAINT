r"""This registry is extended from habitat.Registry to provide
registration for trainer and policies, while keeping Registry
in habitat core intact.

Various decorators for registry different kind of classes with unique keys

-   Register a trainer: ``@baseline_registry.register_trainer``
-   Register a policy: ``@baseline_registry.register_policy``
"""

from typing import Optional
from habitat.core.registry import Registry

class ToponavRegistry(Registry):
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a policy with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.rl.ppo.policy import Policy
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyPolicy(Policy):
                pass


            # or

            @baseline_registry.register_policy(name="MyPolicyName")
            class MyPolicy(Policy):
                pass

        """
        return cls._register_impl("policy", to_register, name)
    
    @classmethod
    def get_policy(cls, name: str):
        r"""Get the policy with :p:`name`."""
        return cls._get_impl("policy", name)
    
    @classmethod
    def register_agent(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        """
        Registers an agent with. Usage:
        ```
        @baseline_registry.register_agent
        class ExampleAgent:
            pass
        ```
        or override the name with `name`.
        ```
        @baseline_registry.register_agent(name="MyAgentAccessMgr")
        class ExampleAgentAccessMgr:
            pass
        ```
        """
        from faint.train.habitat.agents import BaseAgent

        return cls._register_impl(
            "agent", to_register, name, assert_type=BaseAgent
        )

    @classmethod
    def get_agent(cls, name: str):
        return cls._get_impl("agent", name)
    
toponav_registry = ToponavRegistry()

    
