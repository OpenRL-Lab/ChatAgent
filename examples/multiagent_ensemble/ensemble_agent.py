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
import os
import copy
from ChatAgent.agents.base_chat_api_agent import BaseChatAPIAgent
from ChatAgent.vec_agents.sync_vagent import SyncVecAgent
from ChatAgent.agents.zhipu_chat_agent import ZhipuChatAgent
from ChatAgent.agents.dashscope_chat_agent import DashScopeChatAgent
from ChatAgent.utils.request_utils import construct_request_from_text
from custom_agent_wrappers import SummaryWrapper


def GLM4():
    agent = ZhipuChatAgent(model_name="glm-4", api_key=os.getenv("ZHIPU_API_KEY"))
    return agent


def QwenMAX():
    agent = DashScopeChatAgent(model_name="qwen-max", api_key=os.getenv("QWEN_API_KEY"))
    return agent


class EnsembleAgent(BaseChatAPIAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        agent_fns = [GLM4, QwenMAX]
        self.ana_agent_num = len(agent_fns)
        if self.ana_agent_num == 1:
            self.ana_agent = agent_fns[0]()
        else:
            self.ana_agent = SyncVecAgent(agent_fns)
        summary_agent = SummaryWrapper(QwenMAX())
        self.summary_agent = summary_agent

    def chat(self, request):
        request = construct_request_from_text(request)
        obs = request
        if self.ana_agent_num == 1:
            response = self.ana_agent.act(copy.deepcopy(obs))
            ana_answers = [response.choices[0].message.content]
        else:
            responses = self.ana_agent.act([copy.deepcopy(obs) for _ in range(self.ana_agent_num)])
            ana_answers = [response.choices[0].message.content for response in responses]
        response = self.summary_agent.act({
            "request": request,
            "ana_answers": ana_answers
        })
        response = response.choices[0].message.content
        return response
