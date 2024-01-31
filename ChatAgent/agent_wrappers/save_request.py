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
import json
import uuid

from pathlib import Path

from ChatAgent.protocol.openai_api_protocol import ChatCompletionRequest
from ChatAgent.agent_wrappers.base_agent_wrapper import BaseAgentWrapper
from ChatAgent.agents.base_agent import BaseAgent

class SaveRequest(BaseAgentWrapper):
    def __init__(self,agent:BaseAgent, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(agent)

    def act(self, request):

        file_name = str(uuid.uuid4()) + ".json"
        file_path = self.save_dir / file_name
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(request.dict(), file, ensure_ascii=False, indent=4)

        return self.agent.act(request)


if __name__ == '__main__':
    from ChatAgent.agents.repeat_openai_chat_agent import RepeatOpenAIChatAgent as Agent

    agent = Agent()
    agent = SaveRequest(agent, save_dir="./data_saved/")

    # observation = {"prompt":"hi"}

    request = ChatCompletionRequest(
        model='',
        messages=[{'role': 'user', 'content': '你是谁？你叫什么名字？'}],
        # stop=["是"],
    )
    res = agent.act(request)
    print(res)
