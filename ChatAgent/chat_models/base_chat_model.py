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
from typing import Optional


class BaseChatModel(ABC):
    max_concurrent_calls = 100
    name: Optional[str] = ""

    def __init__(self, api_base="", api_key="", model_name="", api_type="open_ai", api_version=None,
                 deployment_name=None):
        self.agent = None
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.deployment_name = deployment_name
        self.api_type = api_type
        self.api_version = api_version
        self.init_agent()
        self.inner_api_key = self._set_check_api()

    def _set_check_api(self) -> Optional[str]:
        return None

    def init_agent(self):
        self.agent = None

    @abstractmethod
    def create_chat_completion(self, request):
        raise NotImplementedError

    def check_api_key(self, token: str) -> bool:
        if self.inner_api_key is None:
            return True
        else:
            return self.inner_api_key == token
