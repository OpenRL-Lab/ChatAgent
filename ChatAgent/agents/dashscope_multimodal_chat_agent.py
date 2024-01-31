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
import os
from typing import List, Dict, Any, Union, Tuple, Optional
from http import HTTPStatus

import tempfile
from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import \
    MultiModalConversationResponse, MultiModalConversationUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from ChatAgent.agents.base_chat_api_agent import BaseChatAPIAgent
from ChatAgent.serve import HTTPException
from ChatAgent.utils.multimodality_tools import save_base64_to_local_image
from ChatAgent.utils.qwen import convert_qwen_to_chat_content, convert_qwen_to_openai_usage, \
    convert_qwen_to_chat_completion


def convert_single_image_url(image_url: str) -> Tuple[str, Optional[str]]:
    local_save_path = None
    if image_url.startswith("data:"):
        local_save_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        save_base64_to_local_image(image_url, save_path=local_save_path)
        image_url = f"file://{local_save_path}"

    return image_url, local_save_path


def convert_image_url(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    local_image_paths = []
    for i, message in enumerate(messages):
        if "content" in message:
            for j, content_item in enumerate(message["content"]):

                if "image" in content_item:
                    image_url = content_item["image"]
                    image_url, local_save_path = convert_single_image_url(image_url)
                    if local_save_path:
                        local_image_paths.append(local_save_path)
                    content_item["image"] = image_url

                message["content"][j] = content_item
            messages[i] = message

    return messages, local_image_paths


def convert_openai2qwen_message(messages: List[Dict[str, Any]]):
    for i, message in enumerate(messages):
        if "content" in message:
            for j, content_item in enumerate(message["content"]):
                new_content_item = {}
                if "image_url" in content_item:
                    image_url = content_item["image_url"]["url"]
                    new_content_item["image"] = image_url
                elif "text" in content_item:
                    new_content_item["text"] = content_item["text"]
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"content中的item只能且必须包含image或者text，但是得到字段: {content_item}"
                    )
                message["content"][j] = new_content_item
            messages[i] = message
    return messages


class DashScopeMultiModalChatAgent(BaseChatAPIAgent):

    def chat(self, request):
        if request.stream:
            raise HTTPException(
                status_code=404,
                detail="千问模型目前不支持stream格式调用"
            )
        messages = convert_openai2qwen_message(request.messages)
        messages, local_image_paths = convert_image_url(messages)
        response = MultiModalConversation.call(api_key=self.api_key, model='qwen-vl-plus',
                                               messages=messages, stream=False)

        for local_image_path in local_image_paths:
            os.remove(local_image_path)
        if response.status_code == HTTPStatus.OK:
            return convert_qwen_to_chat_completion(response, model='qwen-vl-plus')
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.message
            )

# def convert_qwen2openai_message(messages: List[Dict[str, Any]]):
#     for i, message in enumerate(messages):
#         if "content" in message:
#             for j, content_item in enumerate(message["content"]):
#                 new_content_item = {}
#                 if "image" in content_item:
#                     image_url = content_item["image"]
#                     new_content_item["type"] = "image_url"
#                     new_content_item["image_url"] = {"url": image_url}
#                 elif "text" in content_item:
#                     text = content_item["text"]
#                     new_content_item["type"] = "text"
#                     new_content_item["text"] = text
#                 else:
#                     raise HTTPException(
#                         status_code=404,
#                         detail=f"content中的item只能且必须包含image或者text，但是得到字段: {content_item}"
#                     )
#                 message["content"][j] = new_content_item
#             messages[i] = message
#     return messages


# def convert_qwen_to_chat_content(content: Union[str, List]) -> str:
#     if isinstance(content, str):
#         return content
#     if isinstance(content, list):
#         final_str = ""
#         for c in content:
#             if isinstance(c, str):
#                 final_str += c
#             elif isinstance(c, dict):
#                 if "text" in c:
#                     final_str += c["text"]
#
#         return final_str


# def convert_qwen_to_openai_usage(usage: MultiModalConversationUsage) -> CompletionUsage:
#     input_tokens = usage.input_tokens
#     output_tokens = usage.output_tokens
#     return CompletionUsage(
#         prompt_tokens=input_tokens,
#         completion_tokens=output_tokens,
#         total_tokens=input_tokens + output_tokens
#     )


# def convert_qwen_to_chat_completion(response: MultiModalConversationResponse, model: str) -> ChatCompletion:
#     output = response.output
#
#     converted_choices = []
#     for choice in output.choices:
#         content = convert_qwen_to_chat_content(choice.message.content)
#         converted_choice = Choice(
#             finish_reason=choice.finish_reason if choice.finish_reason else "stop",
#             index=0,
#             message=ChatCompletionMessage(
#                 content=content,
#                 role=choice.message.role,
#             ),
#         )
#
#         converted_choices.append(converted_choice)
#
#     return ChatCompletion(
#         id=response.request_id,
#         choices=converted_choices,
#         created=0,
#         model=model,
#         object="chat.completion",
#         usage=convert_qwen_to_openai_usage(response.usage)
#     )
