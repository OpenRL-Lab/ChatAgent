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
from ChatAgent.agents.base_chat_api_agent import BaseChatAPIAgent
from ChatAgent.agents.dashscope_chat_agent import DashScopeChatAgent
from ChatAgent.utils.request_utils import construct_request_from_text

from custom_agent_wrappers import SummaryWrapper
from utils import get_weather

def QwenMAX():
    agent = DashScopeChatAgent(model_name="qwen-max", api_key=os.getenv("QWEN_API_KEY"))
    return agent


class RAGAgent(BaseChatAPIAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        summary_agent = SummaryWrapper(QwenMAX())
        self.summary_agent = summary_agent

    def chat(self, request):
        retrieved_source = "http://www.weather.com.cn/weather1d/101010100.shtml"
        retrieved_answer = get_weather(retrieved_source)

        request = construct_request_from_text(request)
        response = self.summary_agent.act({
            "request": request,
            "retrieved_source": retrieved_source,
            "retrieved_answer": retrieved_answer
        })
        response = response.choices[0].message.content
        return response
