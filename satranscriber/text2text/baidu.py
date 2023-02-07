# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5

from .baidu_apikey import APP_ID, APP_KEY
from typing import Union, List

def make_md5(s: str, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

class Translator:
    def __init__(self, source_lang="ja", target_lang="zh") -> None:
          self.source_lang = source_lang
          self.target_lang = target_lang
          self.query_list = list()

    def translate_request(self, query: str) -> str:
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        salt = random.randint(32768, 65536)
        payload = {
            'appid': APP_ID, 
            'q': query, 
            'from': self.source_lang, 
            'to': self.target_lang, 
            'salt': salt,  
            'sign': make_md5(APP_ID + query + str(salt) + APP_KEY),
        }
        url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        r = requests.post(url, params=payload, headers=headers)
        if r.status_code != 200:
            print(r.content)
            r.raise_for_status()

        # print(r.json())
        return r.json().get("trans_result")
    
    def translate(self, query: str) -> str:
        if query.strip() == "":
            return ""
        self.query_list.append(query)
        self.query_list = self.query_list[-min(10, len(self.query_list)):]
        result = self.translate_request("\n".join(self.query_list))
        return result[-1]["dst"]