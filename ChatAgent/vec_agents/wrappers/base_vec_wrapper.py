#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

from abc import ABC, abstractmethod

from gymnasium.utils import seeding

from ChatAgent.vec_agents.base_vagent import BaseVecAgent

class VecAgentWrapper(BaseVecAgent,ABC):
    def __init__(self,agent:BaseVecAgent):
        self.agent = agent
        self._parallel_agent_num = self.agent.parallel_agent_num

    @property
    def parallel_agent_num(self) -> int:
        return self._parallel_agent_num


    def act(self, observations, *args, **kwargs):
        """Step all environments."""
        return self.agent.act(observations, *args, **kwargs)

    @property
    def unwrapped(self):
        return self.agent.unwrapped

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.agent, name)

    def __del__(self):
        self.agent.__del__()

    def close(self, **kwargs):
        return self.agent.close(**kwargs)

    def close_extras(self, **kwargs):
        return self.agent.close_extras(**kwargs)

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def agent_is_wrapped(
            self, wrapper_class, indices: VecEnvIndices
    ) -> List[bool]:
        return self.agent.agent_is_wrapped(wrapper_class, indices=indices)