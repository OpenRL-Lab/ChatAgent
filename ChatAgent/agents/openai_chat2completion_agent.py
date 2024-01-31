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
from typing import Callable
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types import Completion
from openai.types.chat.chat_completion import ChatCompletion, Choice,ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice


from ChatAgent.agents.base_chat_api_agent import BaseChatAPIAgent


def convert_completion_to_chat_completion(completion: Completion) -> ChatCompletion:
    converted_choices = [Choice(
        finish_reason=choice.finish_reason,
        index=choice.index,
        message = ChatCompletionMessage(
            content=choice.text,
            role="assistant",
        ),
        logprobs=choice.logprobs,
    ) for choice in completion.choices]
    # print("completion:",completion)
    return ChatCompletion(
        id=completion.id,
        choices=converted_choices,
        created=completion.created,
        model=completion.model,
        object="chat.completion",
        system_fingerprint=completion.system_fingerprint,
        usage=completion.usage
    )


def convert_completion_to_chat_chunk(completion: Completion) -> ChatCompletionChunk:
    converted_choices = [ChunkChoice(
        finish_reason=choice.finish_reason,
        index=choice.index,
        delta = ChoiceDelta(
            content=choice.text,
            role="assistant",
        ),
        logprobs=choice.logprobs,
    ) for choice in completion.choices]
    return ChatCompletionChunk(
        id=completion.id,
        choices=converted_choices,
        created=completion.created,
        model=completion.model,
        object="chat.completion.chunk",
        system_fingerprint=completion.system_fingerprint,
    )

class OpenAIChat2CompletionAgent(BaseChatAPIAgent):
    def __init__(self, chat2completion_template: Callable, **kwargs):
        self.chat2completion_template = chat2completion_template
        super().__init__(**kwargs)

    def chat(self, request):
        if self.client is None:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        temperature = request.temperature

        max_tokens = request.max_tokens
        if self.max_tokens is not None:
            if max_tokens is None:
                max_tokens = self.max_tokens
            else:
                max_tokens= min(max_tokens,self.max_tokens)

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

        prompt = self.chat2completion_template(request)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            stop=stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=request.stream,
        )

        if request.stream:
            def stream_generator(stream):
                for chunk in stream:
                    chunk = convert_completion_to_chat_chunk(chunk)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            generator = stream_generator(response)
            return StreamingResponse(generator,
                                     media_type="text/event-stream")
        else:
            response = convert_completion_to_chat_completion(response)

        return response
