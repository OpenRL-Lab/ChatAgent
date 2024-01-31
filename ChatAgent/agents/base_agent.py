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
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

import numpy as np
from gymnasium.utils import seeding

class BaseAgent(ABC):
    _np_random: Optional[np.random.Generator] = None

    def __init__(self):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    @abstractmethod
    def act(self,observation):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the agent's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value


    def __enter__(self):
        """Support with-statement for the agent."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the agent and closes the agent."""
        self.close()
        # propagate exception
        return False