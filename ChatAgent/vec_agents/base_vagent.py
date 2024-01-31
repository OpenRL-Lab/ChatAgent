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
from typing import List, Iterable, Optional

import numpy as np

class BaseVecAgent(ABC):
    closed = False
    _np_random: Optional[np.random.Generator] = None
    parallel_agent_num: int

    def __init__(self,parallel_agent_num: int):
        self.parallel_agent_num = parallel_agent_num
        self.closed = False

    def act(self,observations):
        return self._act(observations)
    @abstractmethod
    def _act(self,observations):
        raise NotImplementedError

    def close(self, **kwargs) -> None:
        if self.closed:
            return
        self.close_extras(**kwargs)
        self.closed = True

    @property
    def unwrapped(self) -> "BaseVecAgent":
        return self

    def agent_is_wrapped(
        self, wrapper_class, indices
    ) -> List[bool]:
        """Check if worker agents are wrapped with a given wrapper"""
        indices = self._get_indices(indices)
        # TODO
        return [False for i in indices]

    def _get_indices(self, indices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.parallel_agent_num)
        elif isinstance(indices, int):
            indices = [indices]
        return indices