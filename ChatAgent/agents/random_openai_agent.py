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

from typing import Union, Optional, List
from abc import ABC, abstractmethod
import random

from openai.types.chat.chat_completion import ChatCompletion,Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from ChatAgent.agents.base_agent import BaseAgent
from ChatAgent.utils.response_utils import construct_response_from_text

class RandomOpenAIAgent(BaseAgent, ABC):
    def __init__(self, choices: List[str]):
        self.choices = choices
        super().__init__()

    def act(self, request):
        return construct_response_from_text(text=self.np_random.choice(self.choices))
