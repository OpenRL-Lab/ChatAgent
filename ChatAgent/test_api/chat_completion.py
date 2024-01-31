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


import openai

def get_chat_response(api_base="",api_key="sk-",model_name="", messages=[], stop_tokens=None,):
    openai.api_base = api_base
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        stop=stop_tokens,
        temperature=0.,
        max_tokens=1000,
    )
    response = completion['choices'][0]['message']['content']
    # print(f"Response:\n {response}")
    return  response
