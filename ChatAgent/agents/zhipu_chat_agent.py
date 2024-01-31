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
from http import HTTPStatus


import zhipuai
from zhipuai import ZhipuAI
from zhipuai.core._base_type import NOT_GIVEN
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from ChatAgent.agents.base_chat_api_agent import BaseChatAPIAgent
from ChatAgent.serve import HTTPException



def convert_zhipu_to_openai_usage(usage: dict) -> CompletionUsage:
    return CompletionUsage(**usage)


def convert_zhipu_to_chat_completion(response: dict, model: str) -> ChatCompletion:
    output = response["data"]

    converted_choices = []
    for choice in output["choices"]:
        converted_choice = Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content=choice["content"],
                role=choice["role"],
            ),
            logprobs=None,
        )

        converted_choices.append(converted_choice)

    return ChatCompletion(
        id=output["task_id"],
        choices=converted_choices,
        created=0,
        model=model,
        object="chat.completion",
        usage=convert_zhipu_to_openai_usage(output["usage"])
    )


class ZhipuChatAgent(BaseChatAPIAgent):

    def chat(self, request):
        if request.stream:
            print("Model:{} 不支持stream调用".format(self.model_name))
            raise HTTPException(
                status_code=404,
                detail="智谱模型目前不支持stream格式调用"
            )

        # zhipuai.api_key = self.api_key
        # response = 	zhipuai.model_api.invoke(model=self.model_name,
        #                            prompt=request.messages, stream=False, temperature=request.temperature,
        #                            top_p=request.top_p,
        #                            )

        client = ZhipuAI(api_key=self.api_key)
        temperature = NOT_GIVEN if request.temperature is None else max(request.temperature, 0.01)

        if temperature != NOT_GIVEN:
            top_p= NOT_GIVEN
        else:
            top_p = NOT_GIVEN if request.top_p is None else max(min(request.top_p, 0.999),0.01)
        assert temperature == NOT_GIVEN or top_p== NOT_GIVEN, "temperature 和 top_p 不能同时设置"

        response = client.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型名称
            messages=request.messages,
            stream=False,
            temperature=temperature,
            top_p=top_p,
        )
        return response

        # if response["code"] == HTTPStatus.OK:
        #     return convert_zhipu_to_chat_completion(response, model=self.model_name)
        # else:
        #     print("Model: {} Error:{}".format(self.model_name, response["msg"]))
        #     raise HTTPException(
        #         status_code=response["code"],
        #         detail=response["msg"]
        #     )
