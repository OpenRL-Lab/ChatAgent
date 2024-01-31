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
from typing import Union, Optional, List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from gymnasium.utils import seeding

from ChatAgent.agents.base_agent import BaseAgent

class BaseAgentWrapper(BaseAgent):

    def __init__(self,agent:BaseAgent):
        self.agent = agent

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self.agent.reset(seed=seed, options=options)


    def act(self,observation):
        return self.agent.act(observation)

    def close(self):
        self.agent.close()

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the :attr:`Agent` :attr:`np_random` attribute."""
        return self.agent.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self.agent.np_random = value

    @property
    def _np_random(self):
        """This code will never be run due to __getattr__ being called prior this.

        It seems that @property overwrites the variable (`_np_random`) meaning that __getattr__ gets called with the missing variable.
        """
        raise AttributeError(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        )