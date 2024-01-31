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
from datetime import datetime
import time

from openai.types.chat.chat_completion import ChatCompletion
from ChatAgent.agents.openai_chat_agent import OpenAIChatAgent

class RepeatOpenAIChatAgent(OpenAIChatAgent):
    def chat(self, request):
        created_timestamp = int(
            time.mktime(datetime.strptime("2021-08-15T15:54:19.563Z", "%Y-%m-%dT%H:%M:%S.%fZ").timetuple()))

        response = ChatCompletion(
            choices=[
                {
                    "finish_reason":"stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content":request.messages[0]["content"],
                        "role": "assistant",  # 根据定义填写角色
                        # 如果需要，可以添加 function_call 和 tool_calls 字段
                    },
                }
            ],
            created=created_timestamp,  # 使用 Unix 时间戳
            id="cmpl-3G9XqYRZJzXJ0YyZq3yvK6Jp",
            model="davinci:2020-05-03",
            object="chat.completion",  # 更正字段值
        )

        return response
