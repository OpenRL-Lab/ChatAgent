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
from typing import Optional, Dict, Any, Tuple, List
import copy

from ChatAgent.agent_wrappers.base_agent_wrapper import BaseAgentWrapper
from ChatAgent.agents.base_agent import BaseAgent
from ChatAgent.utils.response_utils import construct_response_from_text


class RetryInChoices(BaseAgentWrapper):
    def __init__(self, agent: BaseAgent, choices: List[str], retry_times: int = 3):
        assert choices, "choices should not be empty"
        assert retry_times > 0, "retry_times should be positive"

        self.choices = choices
        self.retry_times = retry_times

        super().__init__(agent)

    def act(self, request):
        original_request = copy.deepcopy(request)
        retry_time = 0
        wrong_texts = []
        response = None
        while retry_time < self.retry_times:
            retry_time += 1
            if wrong_texts:
                request = self.convert_request(copy.deepcopy(original_request), wrong_texts)
            response = self.agent.act(request)

            if self.good_answer(response, original_request):
                return self.polish_response(response)

            wrong_text = response.choices[0].message.content
            wrong_texts.append(wrong_text)
            print(f"Need {retry_time + 1}th retry, because of wrong answer: {wrong_text}.")
        # return self.get_random_response(original_request)
        return response
    def get_random_response(self, request):
        return construct_response_from_text(text=self.np_random.choice(self.choices))

    def convert_request(self, request, wrong_texts: List[str]):
        raise NotImplementedError

    def good_answer(self, response, request) -> bool:
        text = response.choices[0].message.content
        text = self.polish_answer(text)
        return text in self.choices

    def polish_answer(self, text: str):
        return (text.
                replace(",", "").
                replace(";", "").
                replace(".", "").
                replace(" ", "").
                replace("\n", "").
                replace("。", "").
                replace("，", "").
                replace("；", "").
                replace("和", ""))

    def polish_response(self, response):
        text = response.choices[0].message.content
        response.choices[0].message.content = self.polish_answer(text)
        return response
