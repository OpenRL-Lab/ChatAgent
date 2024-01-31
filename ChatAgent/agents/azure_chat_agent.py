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
from fastapi.responses import StreamingResponse
from openai import AzureOpenAI

from ChatAgent.agents.base_chat_api_agent import BaseChatAPIAgent


class AzureChatAgent(BaseChatAPIAgent):
    def chat(self, request):
        if self.client is None:
            self.client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_base,api_version="2023-12-01-preview",)

        messages = request.messages
        temperature = request.temperature

        max_tokens = request.max_tokens
        if self.max_tokens is not None:
            if max_tokens is None:
                max_tokens = self.max_tokens
            else:
                max_tokens = min(max_tokens, self.max_tokens)

        stop = request.stop
        stop_tokens = self.stop_tokens
        if stop is not None:
            if isinstance(stop, str):
                stop_tokens.append(stop)
            if isinstance(stop, list):
                stop_tokens += stop
        if stop_tokens == []:
            stop_tokens = None
        else:
            stop_tokens = list(set(stop_tokens))


        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=request.stream,
        )

        if request.stream:
            def stream_generator(stream):
                for chunk in stream:
                    if chunk.choices is None or chunk.choices == []:
                        continue
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            generator = stream_generator(response)
            return StreamingResponse(generator,
                                     media_type="text/event-stream")

        return response
