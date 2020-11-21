from random import random
import json
from gensim.models import Word2Vec as w2v
import os
from gensim.models.word2vec import LineSentence

config_path = "globel_config.json"


# 加载全局配置文件
def Load_Configs():
    with open(config_path,encoding="utf8") as f:
        configs = json.load(f)
    return configs
