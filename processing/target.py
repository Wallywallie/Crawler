from simhash import Simhash

import requests
import json
"""
处理收集的txt demo
"""

import re
import os

import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


API_KEY = "sXu3ML0Q1VitaW9X98Zo2yJQ"
SECRET_KEY = "pYEw0w3kBLbBT8CDLqZR0NpoNbxmXzi4"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    response = requests.post(url, params=params)
    return response.json().get("access_token")

def get_assistant_reply(user_message):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k?access_token=" + get_access_token()
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": user_message,
                "key": "1"
            }
        ],
        "temperature": 0.95,
        "top_p": 0.7,
        "penalty_score": 1,
        "collapsed": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    result = response.json()

    assistant_message = result['result'] if 'result' in result and len(result['result']) > 1 else "Error retrieving response"
    return assistant_message


#使用并查集来记录pair
class UnionFind:
    def __init__(self, n):
        # 初始化并查集，每个节点的父节点初始化为自己
        self.parent = list(range(n))
        self.sz = n

    def find(self, x):
        # 查找集合根节点，并进行路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # 合并两个节点所在的集合
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

    def get_groups(self) -> dict:
        groups = {}
        for i in range(self.sz):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups

class Fussion:
    def __init__(self):
        # 加载spaCy的语言模型，用于文本向量化
        self.nlp = spacy.load('zh_core_web_md')  # 你也可以使用其他语言模型

    def split_paragraphs(self, text):
        """
        将文本按自然段划分
        """
        return [para.strip() for para in text.split('\n') if para.strip()]
    
    def vectorize_paragraphs(self, paragraphs):
        """
        将每个段落向量化
        """
        return [self.nlp(para).vector for para in paragraphs]
    
    def find_matching(self, vectors1, vectors2, threshold=0.75):
        """
        通过余弦相似度匹配两个文本的段落
        """
        matches = []
        for i, vec1 in enumerate(vectors1):
            best_match = -1
            best_similarity = threshold
            for j, vec2 in enumerate(vectors2):
                similarity = cosine_similarity([vec1], [vec2])[0][0]
                if similarity > best_similarity:
                    best_match = j
                    best_similarity = similarity
            matches.append((i, best_match, best_similarity))
        print(matches)    
        return matches
    def find_best_match(self, vec1, vectors2, threshold=0.75):
        """
        寻找与 vec1 最相似的段落，并返回其索引和相似度
        """
        best_match = -1
        best_similarity = threshold
        for j, vec2 in enumerate(vectors2):
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            if similarity > best_similarity:
                best_match = j
                best_similarity = similarity
        return best_match, best_similarity    

    def merge_paragraphs_llm(self, paragraphs1, paragraphs2, matches):
        """
        使用大语言模型根据匹配结果合并两个文本的段落
        """

               
        merged_text = []
        llm_text = []
        matched_indices = set()
        
        for i, j, sim in matches:
            if j != -1:
                merged_text.append(f"段落 {i+1} (匹配):\n{paragraphs1[i]}\n---\n{paragraphs2[j]}")
                matched_indices.add(j)
                prompt = f"""
                            ##ROLE##你是一个擅长文本融合的nlp专家，禁止回复文本以外的内容
                            ##TASK##你的任务是把两段内容相近文本合成一段文本
                            ##RESTRICTION##尽可能不要改变文本的语序，不要删除语义信息,禁止回复文本以外的内容，禁止解释，禁止注释
                            ##文本一##{paragraphs1[i]}
                            ##文本二##{paragraphs2[j]}
                            禁止回复文本以外的内容
                            """                 
                llm_res = get_assistant_reply(prompt)
                llm_text.append(llm_res)
            else:
                merged_text.append(f"段落 {i+1} (未匹配):\n{paragraphs1[i]}")
                llm_text.append(paragraphs1[i])

        
        # 处理第二个文本中未匹配的段落
        for k in range(len(paragraphs2)):
            if k not in matched_indices:
                merged_text.append(f"段落 {len(paragraphs1) + k + 1} (额外):\n{paragraphs2[k]}")
                llm_text.append(paragraphs2[k])
        

        
        return '\n'.join(merged_text),'\n'.join(llm_text)



    def merge_paragraphs(self, paragraphs1, paragraphs2,vectors1, vectors2, threshold):
        """
        根据匹配结果合并两个文本的段落，未匹配段落保持原顺序
        以段落短的为基准，目前考虑两种情况：匹配上-未匹配上，需要综合考虑匹配程度和步数来判断是否要在当前匹配
        允许多端匹配一段，但必须按顺序匹配

        """

        merged_text = []
        used_indices2 = set()
        i, j = 0, 0

        while i < len(paragraphs1) or j < len(paragraphs2):
            # 处理第一个文本的段落
            if i < len(paragraphs1):
                match_index, similarity = self.find_best_match(vectors1[i], vectors2, threshold)
                
                if match_index != -1 and match_index not in used_indices2:
                    # 如果找到匹配
                    merged_text.append(f"段落 {i+1} (匹配):\n{paragraphs1[i]}\n---\n{paragraphs2[match_index]}")
                    used_indices2.add(match_index)
                    j = match_index + 1  # 确保下一个j从match_index之后开始
                else:
                    # 没有找到匹配，直接保留段落1
                    merged_text.append(f"段落 {i+1} (未匹配):\n{paragraphs1[i]}")
                i += 1

            # 处理第二个文本的段落
            if j < len(paragraphs2):
                if j not in used_indices2:
                    # 如果段落2没有被匹配，直接保留
                    merged_text.append(f"段落 {len(paragraphs1) + j + 1} (未匹配):\n{paragraphs2[j]}")
                j += 1
        
        return '\n\n'.join(merged_text)
    
    def fussion(self, text1, text2, threshold=0.75):
        """
        将两个文本段落融合
        """
        # 1. 划分段落
        paragraphs1 = self.split_paragraphs(text1)
        paragraphs2 = self.split_paragraphs(text2)
        
        # 2. 向量化段落
        vectors1 = self.vectorize_paragraphs(paragraphs1)
        vectors2 = self.vectorize_paragraphs(paragraphs2)
        

        # 3. 语义匹配
        matches = self.find_matching(vectors1, vectors2)
        
        # 4. 合并段落
        merged_text, llm_text = self.merge_paragraphs_llm(paragraphs1, paragraphs2, matches)
        
        return merged_text, llm_text



#1.识别哪些文本是相似的

# 假设文本文件在这个文件夹中
folder_path = 'D:\CS\crawler\data'
texts = []
paras = []

def remove_long_english(text : str) -> str:


    pattern = r'([a-zA-Z0-9\'\".,!?;:()\s]){20,}'

    # 去除多余的空格
    cleaned_text = re.sub(pattern, '', text)


    cleaned_text = re.sub(r'^▼.*\n?', '', cleaned_text, flags=re.MULTILINE)

    return cleaned_text

def remove_non_chinese(text):
    # 使用正则表达式匹配所有非汉字字符并替换为空字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fff]+', '', text)
    return cleaned_text

"""
path = "D:\CS\crawler\data\叠层历险记：尼康上海直营店 _ 芝作室 - 谷德设计网.txt"
with open(path, 'r', encoding= 'utf-8') as file:
    txt = file.read()
    print(txt)
    print("=======================")
    print(remove_non_chinese(txt))
"""

# 读取所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            txt = file.read()
            paras.append(txt)
            text = remove_non_chinese(txt)
            texts.append(text)
            

# 计算SimHash值
hashes = [Simhash(text) for text in texts]


def is_similar(hash1, hash2, threshold=3):
    return hash1.distance(hash2) <= threshold


# 检查所有文本对
similar_pairs = []
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        dis = hashes[i].distance(hashes[j])
        if (dis < 25):
            print(f"Text{i}-Text{j}:{dis}")  
            similar_pairs.append((i, j))





uf = UnionFind(len(paras))

for x, y in similar_pairs:
    uf.union(x, y)

# 获取并输出相似文本组
similar_groups = uf.get_groups()
print(similar_groups)





text1 = remove_long_english(paras[similar_groups[3][1]])

text2 = remove_long_english(paras[similar_groups[3][0]])




fussion_tool = Fussion()
merged_text,llm_text = fussion_tool.fussion(text1, text2)
print("======================================")
print(merged_text)
print("======================================")
print(llm_text)
print("======================================")