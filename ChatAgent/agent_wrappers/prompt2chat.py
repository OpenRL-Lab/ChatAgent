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
from typing import Dict, Any

from ChatAgent.protocol.openai_api_protocol import ChatCompletionRequest
from ChatAgent.agent_wrappers.base_agent_wrapper import BaseAgentWrapper


class Prompt2Chat(BaseAgentWrapper):
    def act(self, observation: Dict[str, Any]):
        prompt = observation["prompt"]
        request = observation.pop("request", None)
        if request is None:
            request = ChatCompletionRequest(
                model='',
                messages=[{'role': 'user', 'content': prompt}],
            )
        else:
            request.messages = [{'role': 'user', 'content': prompt}]
        return self.agent.act(request)


if __name__ == '__main__':
    from ChatAgent.agents.repeat_openai_chat_agent import RepeatOpenAIChatAgent as Agent

    agent = Agent()
    agent = Prompt2Chat(agent)

    # observation = {"prompt":"hi"}

    request = ChatCompletionRequest(
        model='',
        messages=[{'role': 'user', 'content': '你是谁？你叫什么名字？'}],
        # stop=["是"],
    )
    observation = {"prompt": "hi", "request": request}

    res = agent.act(observation)
    print(res)
