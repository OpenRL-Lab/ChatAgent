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
from typing import Union, Tuple, Dict

import json
from pathlib import Path

from ChatAgent.protocol.openai_api_protocol import MultimodalityChatCompletionRequest, ChatCompletionRequest


def construct_request_from_text(text: str)->ChatCompletionRequest:
    data = {
        "model": "",
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": -1,
        "n": 1,
        "max_tokens": 1000,
        "stop": None,
        "stream": False,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "user": None,
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ]
        }
    return ChatCompletionRequest(**data)


def get_MultimodalityChatCompletionRequest(data: Dict) -> MultimodalityChatCompletionRequest:
    new_data = {
        "model": "",
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": -1,
        "n": 1,
        "max_tokens": 1000,
        "stop": None,
        "stream": False,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "user": None,
        "messages": [
            {
                "role": "user",
                "content": data["prompt"]
            }
        ]
        }

    return MultimodalityChatCompletionRequest(**new_data)


def get_ChatCompletionRequest(data: dict) -> ChatCompletionRequest:
    raise NotImplementedError


def load_request_from_case_json(json_file: Union[str, Path], data_type: str = "ChatCompletionRequest") -> Tuple[Union[
    MultimodalityChatCompletionRequest, ChatCompletionRequest], str]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    answer = data["answer"]
    if data_type == "MultimodalityChatCompletionRequest":
        request = get_MultimodalityChatCompletionRequest(data)
    elif data_type == "ChatCompletionRequest":
        request = get_ChatCompletionRequest(data)
    else:
        raise ValueError(f"Unknown data_type {data_type}")

    return request, answer


def load_img_and_text_from_json(json_file: Union[str, Path]) -> Tuple[str, str]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    messages = data["messages"]
    for message in messages:
        if message["role"] == "user":
            contents = message["content"]
            for content in contents:
                if content["type"] == "text":
                    text = content["text"]
                if content["type"] == "image_url":
                    img = content["image_url"]["url"]

    return img, text


def load_request_from_json(json_file: Union[str, Path], data_type: str = "ChatCompletionRequest") -> Union[
    MultimodalityChatCompletionRequest, ChatCompletionRequest]:
    with open(json_file, 'r') as f:
        data = json.load(f)

    if data_type == "MultimodalityChatCompletionRequest":
        return MultimodalityChatCompletionRequest(**data)
    elif data_type == "ChatCompletionRequest":
        return ChatCompletionRequest(**data)
    else:
        raise ValueError(f"Unknown data_type {data_type}")


def load_img_and_text_from_json(json_file: Union[str, Path]) -> Tuple[str, str]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    messages = data["messages"]
    for message in messages:
        if message["role"] == "user":
            contents = message["content"]
            for content in contents:
                if content["type"] == "text":
                    text = content["text"]
                if content["type"] == "image_url":
                    img = content["image_url"]["url"]

    return img, text
