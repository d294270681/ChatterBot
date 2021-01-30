'''
语料库预处理
'''

import os
import jieba
from tkinter import _flatten

#读取数据
file_path = './dialog'
file_names = os.listdir(file_path)
file_name = file_names[0]
corpus = []
for file_name in file_names:
    with open(os.path.join(file_path,file_name),'r',encoding='utf-8') as f:
        corpus.extend(f.readlines())
print(corpus)

#中文分词
jieba.load_userdict('./mydict.txt')
corpus_cut = [jieba.lcut(i.replace('\n', '')) for i in corpus]

#构建词典
all_dict = list(set(_flatten(corpus_cut)))

#储存词典和分词后的语料文件到指定目录
save_path = './ids'
if not os.path.exists(save_path):
    os.mkdir(save_path)
source = corpus_cut[::2]
target = corpus_cut[1::2]

with open(os.path.join(save_path, 'all_dict.txt'), 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(all_dict))
with open(os.path.join(save_path, 'source.txt'), 'w', encoding='utf-8') as f:
    f.writelines([' '.join(i) +'\n' for i in source])
with open(os.path.join(save_path, 'target.txt'), 'w', encoding='utf-8') as f:
    f.writelines([' '.join(i) +'\n' for i in target])