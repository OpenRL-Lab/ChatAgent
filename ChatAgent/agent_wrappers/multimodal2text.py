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
from typing import Optional, Dict, Any, Tuple, List, Union
import copy

from ChatAgent.agent_wrappers.base_agent_wrapper import BaseAgentWrapper
from ChatAgent.agents.base_agent import BaseAgent
from ChatAgent.protocol.openai_api_protocol import MultimodalityChatCompletionRequest, ChatCompletionRequest


# Convert the multimodal request to pure language-based request
class MultiModal2Text(BaseAgentWrapper):
    def act(self, request: MultimodalityChatCompletionRequest):
        request = self._convert_request(request)
        return self.agent.act(request)

    def _convert_request(self, request: MultimodalityChatCompletionRequest) -> ChatCompletionRequest:
        new_request = copy.deepcopy(request)
        new_request.messages = []
        for messgae in request.messages:
            new_request.messages.append(self._convert_message(messgae))
        return new_request


    def _convert_message(self, message_item: Dict[str, Any]) -> Dict[str, str]:
        if message_item["role"] == "user":
            if isinstance(message_item["content"], str):
                return message_item
            elif isinstance(message_item["content"], list):
                text = ""
                for content in message_item["content"]:
                    if content["type"] == "text":
                        text += content["text"]
                new_message_item = copy.deepcopy(message_item)
                new_message_item["content"] = text
                return new_message_item
            else:
                raise ValueError(f"Unknown content type {type(message_item['content'])}")

        else:
            return message_item
