import os
from typing import (List)


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
    

def get_title(folder_name:str) -> List[str]:
    """
    获取子文件夹下所有文件名
    """
    current_path = os.getcwd()
    subfolder_path = os.path.join(current_path, folder_name)
    filenames = []
    if not os.path.exists(subfolder_path):
        return filenames
    for filename in os.listdir(subfolder_path):
        # 检查是否是文件而不是文件夹
        if os.path.isfile(os.path.join(subfolder_path, filename)):
            # 去掉扩展名并添加到列表
            filenames.append(os.path.splitext(filename)[0])
    return filenames
    
    
