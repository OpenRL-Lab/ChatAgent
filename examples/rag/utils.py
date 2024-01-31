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

from bs4 import BeautifulSoup
import requests


def get_weather(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "lxml")
    wea = soup.select('#today > div.t > ul > li:nth-child(1) > p.wea')
    tem = soup.select('#today > div.t > ul > li:nth-child(1) > p.tem > span')
    weather_info = "今日天气：" + wea[0].string + "。" + "今日温度：" + tem[0].string + "摄氏度。"
    return weather_info


if __name__ == '__main__':
    url = "http://www.weather.com.cn/weather1d/101010100.shtml"
    weather = get_weather(url)
    print(weather)
