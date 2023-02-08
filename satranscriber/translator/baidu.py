# -*- coding: utf-8 -*-

# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
from hashlib import md5

from . import translator

BAIDU_API_URL = "http://api.fanyi.baidu.com/api/trans/vip/translate"

def make_md5(s: str, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

class Translator(translator.Translator):
    def authentication(self, app_key, app_secret, **kwargs):
        self.app_id = app_key
        self.app_key = app_secret

    def translate_request(self, query: str) -> str:
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        salt = random.randint(32768, 65536)
        payload = {
            'appid': self.app_id, 
            'q': query, 
            'from': self.source_lang, 
            'to': self.target_lang, 
            'salt': salt,  
            'sign': make_md5(self.app_id + query + str(salt) + self.app_key),
        }
        
        r = requests.post(BAIDU_API_URL, params=payload, headers=headers)

        if r.status_code != 200:
            print(r.content)
            r.raise_for_status()

        return r.json().get("trans_result")
    
    def translate(self, query: str) -> str:
        result = self.translate_request(query)
        return result[-1]["dst"]
