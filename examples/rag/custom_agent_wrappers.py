#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2024 The OpenRL Authors.
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
from typing import Dict, Any

from ChatAgent.agent_wrappers.base_agent_wrapper import BaseAgentWrapper


def convert_text(text: str, obs: Dict[str, Any]) -> str:
    retrieved_source = obs["retrieved_source"]
    retrieved_answer = obs["retrieved_answer"]
    final_text = "你需要回答来自用户的一个请求。我们还会给出数据库查询的结果。\n"
    final_text += "\n来自用户的请求如下：\n#用户问题开始#\n" + text + "\n#用户问题结束#\n"
    final_text += f"\n数据库查询的结果如下：\n#数据库查询结果开始#\n" + retrieved_answer + "\n#数据库查询结果结束#\n"
    final_text += "\n数据库数据来源："+retrieved_source+"\n"
    final_text += "\n请你根据数据库的查询结果，给出你对来自用户的问题的最终答案。"

    return final_text


def convert_message(message: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
    if message["role"] == "user":
        content = message["content"]
        message["content"] = convert_text(content, obs)
    return message


def convert_request(obs: Dict[str, Any]):
    request = obs["request"]
    messages = request.messages
    new_messages = []
    for message in messages:
        new_message = convert_message(message, obs)
        new_messages.append(new_message)
    request.messages = new_messages
    return request


class SummaryWrapper(BaseAgentWrapper):
    def act(self, obs: Dict[str, Any]):
        request = convert_request(obs)
        return self.agent.act(request)
