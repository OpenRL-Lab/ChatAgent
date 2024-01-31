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
    ana_answers = obs["ana_answers"]
    final_text = "你需要回答来自用户的一个请求。我们还会给出其他模型关于这个请求的回答。\n"
    final_text += "\n来自用户的请求如下：\n#用户问题开始#\n" + text + "\n#用户问题结束#\n"
    final_text += f"\n有{len(ana_answers)}个其他模型参与了这个请求的回答，它们的回答如下（它们的回答不一定正确）：\n"
    for i, ana_answer in enumerate(ana_answers):
        final_text += f"- 模型{i + 1}的回答：\n#模型{i + 1}回答开始#\n" + ana_answer + f"\n#模型{i + 1}回答结束#\n"
    final_text += "\n请你根据其他模型的回答，给出你对来自用户的问题的最终答案，给出你自己的答案即可，不用分析其他模型答案的对错。"
    print("\n给总结智能体的文本：\n", final_text)
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
