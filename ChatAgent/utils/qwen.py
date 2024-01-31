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

from typing import List, Union

from dashscope import Generation
from dashscope.api_entities.dashscope_response import GenerationResponse, GenerationUsage
from dashscope.api_entities.dashscope_response import \
    MultiModalConversationResponse, MultiModalConversationUsage

from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

def convert_qwen_to_chat_content(content: Union[str, List]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        final_str = ""
        for c in content:
            if isinstance(c, str):
                final_str += c
            elif isinstance(c, dict):
                if "text" in c:
                    final_str += c["text"]

        return final_str


def convert_qwen_to_openai_usage(usage: Union[MultiModalConversationUsage,GenerationUsage]) -> CompletionUsage:
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    return CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens
    )


def convert_qwen_to_chat_completion(response: Union[GenerationResponse,MultiModalConversationResponse], model: str) -> ChatCompletion:
    output = response.output

    converted_choices = []
    for choice in output.choices:
        content = convert_qwen_to_chat_content(choice.message.content)
        converted_choice = Choice(
            finish_reason=choice.finish_reason if choice.finish_reason else "stop",
            index=0,
            message=ChatCompletionMessage(
                content=content,
                role=choice.message.role,
            ),
            logprobs=None
        )

        converted_choices.append(converted_choice)

    return ChatCompletion(
        id=response.request_id,
        choices=converted_choices,
        created=0,
        model=model,
        object="chat.completion",
        usage=convert_qwen_to_openai_usage(response.usage)
    )

