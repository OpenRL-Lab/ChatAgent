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

from abc import ABC, abstractmethod

import requests

from ChatAgent.agents.base_api_agent import BaseAPIAgent
from ChatAgent.utils.response_utils import construct_response_from_text

class MultimodalQueryAgent(BaseAPIAgent, ABC):
    def complete(self, request):
        contents = request.messages[0]["content"]
        text = None
        image = None
        for content in contents:
            if content["type"] == "text":
                text = content["text"]
            elif content["type"] == "image_url":
                image = content["image_url"]["url"]
        if text is None or image is None:
            raise ValueError("text or image is None")

        assert image.startswith("data:")

        payload = {
            "text": text,
            "image": image,
        }

        response = requests.post(self.api_base, json=payload)
        text = response.json()["text"]
        return construct_response_from_text(text)

