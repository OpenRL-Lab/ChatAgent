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
import os
from ChatAgent import serve
from ChatAgent.chat_models.base_chat_model import BaseChatModel
from ChatAgent.agents.zhipu_chat_agent import ZhipuChatAgent
from ChatAgent.protocol.openai_api_protocol import MultimodalityChatCompletionRequest
class GLM4(BaseChatModel):
    def init_agent(self):
        self.agent = ZhipuChatAgent(model_name="glm-4", api_key=os.getenv("ZHIPU_API_KEY"))
    def create_chat_completion(self, request):
        return self.agent.act(request)
@serve.create_chat_completion()
async def implement_completions(request: MultimodalityChatCompletionRequest):
    return GLM4().create_chat_completion(request)
serve.run(host="0.0.0.0", port=6368)

