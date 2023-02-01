import sys
import uuid
import requests
import hashlib
import time
from imp import reload
from . import youdao_apikey


reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


class translator:
    def __init__(self, source_lang="ja", target_lang="zh_CHS") -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.app_key = youdao_apikey.APP_KEY
        self.app_secret = youdao_apikey.APP_SECRET

    def translate(self, q: str):
        curtime = str(int(time.time()))
        salt = str(uuid.uuid1())

        r = requests.post(YOUDAO_URL,
                        data={
                            "q": q,
                            "from": self.source_lang,
                            "to": self.target_lang,
                            "appKey": self.app_key,
                            "salt": salt,
                            "sign": encrypt(self.app_key + truncate(q) + salt + curtime + self.app_secret),
                            "signType": "v3",
                            "curtime": curtime,
                            "strict": "true",
                            #   "vocabId":
                        },
                        headers={
                            'Content-Type': 'application/x-www-form-urlencoded'
                        })
        if r.json().get("errorCode") == "0":
            return r.json().get("translation")[0]
        else:
            raise r.content
