import uuid
import requests
import hashlib
import time

from . import translator

YOUDAO_API_URL = 'https://openapi.youdao.com/api'

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


class Translator(translator.Translator):

    def authentication(self, app_key, app_secret, **kwargs):
        self.app_key = app_key
        self.app_secret = app_secret

    def translate(self, q: str):
        curtime = str(int(time.time()))
        salt = str(uuid.uuid1())

        r = requests.post(
            YOUDAO_API_URL,
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
            }
        )
        
        if r.json().get("errorCode") == "0":
            text = r.json().get("translation")[0]
            return text.split()[-1]
        else:
            print(r.content)
            raise requests.HTTPError
