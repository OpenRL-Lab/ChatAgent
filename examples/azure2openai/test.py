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
from openai import OpenAI
client = OpenAI(api_key="sk-", base_url="http://127.0.0.1:6369/v1")
response = client.chat.completions.create(
    model="qwen-max",
    messages=[{"role": "user", "content": "你是谁？"}])
response = response.choices[0].message.content
print(f"模型回复是:\n {response}")
