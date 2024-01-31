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

import logging

from rich.logging import RichHandler


class Logger:
    def __init__(
        self,
        log_level: int = logging.DEBUG,
    ) -> None:
        # TODO: change these flags to log_backend

        self.log_level = log_level
        self._init()

    def _init(self) -> None:
        handlers = [RichHandler()]

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=handlers,
        )

    def info(self, msg: str):
        logging.info(msg)
