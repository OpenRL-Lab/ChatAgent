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

from openai.types.chat.chat_completion import ChatCompletion,Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

def construct_response_from_text(text:str,model:str=""):
    choices = [Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(
            content=text,
            role="assistant",
        ),
        logprobs=None,
    )]
    usage = CompletionUsage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0
    )
    response = ChatCompletion(
        id="",
        choices=choices,
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )
    return response