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
from typing import Iterable, Callable
import copy

from ChatAgent.agents.base_agent import BaseAgent
from ChatAgent.vec_agents.base_vagent import BaseVecAgent

class SyncVecAgent(BaseVecAgent):
    def __init__(self,agent_fns: Iterable[Callable[[], BaseAgent]],):
        self.agent_fns = agent_fns
        self.agents = []
        self.agents += [agent_fn() for agent_fn in agent_fns]
        super().__init__(
            parallel_agent_num=len(self.agents),
        )
        self._actions = []

    def _act(self,observations):
        self._actions = [agent.act(observation) for agent,observation in zip(self.agents,observations)]
        return copy.copy(self._actions)

    def close_extras(self, **kwargs):
        for agent in self.agents:
            agent.close()