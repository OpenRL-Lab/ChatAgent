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
from typing import Union
import base64
import io

from pathlib import Path
from PIL import Image


def get_base64(file_path: str):
    local_img = Path(file_path)
    assert local_img.exists(), f"File not found: {file_path}"
    ext = local_img.suffix.lower()

    # 确定 MIME 类型
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    elif ext == '.gif':
        mime_type = 'image/gif'
    else:
        raise ValueError("Unsupported image type")

    # 读取图片文件，并转换为 base64 编码
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # 构建 data URL
    data_url = f"data:{mime_type};base64,{encoded_string}"
    return data_url

def save_base64_to_local_image(base64_str, save_path: Union[str, Path], quality=95):
    if isinstance(save_path, Path):
        save_path = str(save_path)

    # 去掉 base64 数据前的 URL 部分
    base64_data = base64_str.split(',')[1]
    # 将 base64 字符串转换为二进制数据
    img_data = base64.b64decode(base64_data)
    # 将二进制数据转换为图片对象
    img = Image.open(io.BytesIO(img_data))
    # 检查文件扩展名以确定格式
    if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
        img.save(save_path, 'JPEG', quality=quality)
    else:
        img.save(save_path)