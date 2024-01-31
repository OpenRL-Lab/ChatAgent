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
from typing import Callable, Optional, List
from functools import wraps
import threading

from pydantic_settings import BaseSettings
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn

from ChatAgent.protocol.openai_api_protocol import ChatCompletionRequest


class AppSettings(BaseSettings):
    # The address of the model controller.
    api_keys: Optional[List[str]] = None


app_settings = AppSettings()

app = FastAPI(openapi_url=None)


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
        auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


async def get_api_key(
        auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if auth is None:
        return ""
    token = auth.credentials
    return token


def create_chat_completion():
    def decorator(func: Callable):
        @app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
        @wraps(func)
        async def inner(request: ChatCompletionRequest, *args, **kwargs):
            return await func(request, *args, **kwargs)

        return inner

    return decorator


def run(host="127.0.0.1", port=8100, detached=False):
    global server
    config = uvicorn.Config(app, host=host, port=port)
    server = Server(config=config)
    if detached:
        thread = threading.Thread(target=server.run)
        thread.start()
    else:
        server.run()


def stop():
    global server
    if server:
        server.should_exit = True
        server = None
