from typing import (Dict, List)
import re
import cv2 as cv
import os
import numpy as np
import requests
from enum import Enum
from tqdm import tqdm
from sklearn.cluster import DBSCAN
data1 = [
    {"arch_name": "萨和酒店", "arch_loc": "中国，上海，浦东", "architect" : "小红", "arch_com": "東木筑造","built_time": "2023年" , "img": ["https://oss.gooood.cn/uploads/2024/12/049-Narratives-of-Time.jpg"]},
    {"arch_name": "萨西酒店", "arch_loc": "中国，上海，浦东", "architect" : "小刚", "arch_com": "東木筑造","built_time": "2025年12月", "img":["https://oss.gooood.cn/uploads/2024/12/050-Narratives-of-Time.jpg"]},
    {"arch_name": "Tripolis Park 办公园", "arch_loc": "中国，陕西", "architect" : "小红", "arch_com": "西木筑造","built_time": "2025年","img":["https://oss.gooood.cn/uploads/2024/12/049-Narratives-of-Time.jpg","https://oss.gooood.cn/uploads/2024/12/050-Narratives-of-Time.jpg" ] },
    {"arch_name": "Tripolis办公综合体", "arch_loc": "中国，陕西", "architect" : "小红", "arch_com": "西木筑造","built_time": "2025年","img":["https://oss.gooood.cn/uploads/2024/12/050-Narratives-of-Time.jpg"]}
]
data_img = [
{"img": ["https://upload.wikimedia.org/wikipedia/commons/a/aa/Wukang_Mansion_%2820191114161507%29.jpg"]},
{"img": ["https://k.sinaimg.cn/n/sinakd10101/394/w888h1106/20200308/34dd-iqmtvwv8953624.jpg/w700d1q75cms.jpg"]}
]

class Feat(Enum):
    arch_name = "arch_name"
    arch_loc = "arch_loc"
    architect = "architect"
    arch_com = "arch_com"
    built_time = "built_time"  
    img = "img"


class Integration:
    feat = ["arch_name","arch_loc","architect","arch_com","built_time"]
    def __init__(self, data: List[Dict]):
        self.data = data

    def text_similarity(self, text1: str, text2: str):
        def segment(text:str):
            pattern = r'[^\u4e00-\u9fa5]+|[\u4e00-\u9fa5]+'
            segments = re.findall(pattern, text)
            result = []
            for part in segments:
                if re.match(r'[\u4e00-\u9fa5]', part):  # 如果是中文
                    result.extend(list(part))  # 中文逐字分词
                else: 
                    result.extend(part.split())  # 英文按空格分词    
            return result            
            # 文本预处理：分词+去重
        
        set1 = set(segment(text1))  # 简单分词
        set2 = set(segment(text2))
        # 计算交集与并集
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # 计算Jaccard相似度
        similarity = len(intersection) / len(union) if union else 0

        print(similarity)
        return similarity

    def similarity(self, i, j, feature):
        if feature in [Feat.architect, Feat.arch_com]:
            return self.__architect(i, j, feature.value)
        
        if feature == Feat.arch_name:
            return self.__arch_name(i, j)    
            
        if feature == Feat.arch_loc:
            return self.__arch_loc(i, j)
        
        if feature == Feat.built_time:
            return self.__built_time(i,j)
        
        if feature == Feat.img:
            return self.__img(i, j)

        print(f"data {i} and data {j} are not comparable.")

        return 0

    def __architect(self, i, j, feature):
        arch_i = list(map(str.strip, self.data[i][feature].split("，")))
        arch_j = list(map(str.strip, self.data[j][feature].split("，")))


        for i in range(len(arch_i)):
            for j in range(len(arch_j)):
                if arch_i == arch_j :
                    return 1
        return 0

    def __arch_name(self, i, j):
        name_i = list(map(str.strip, self.data[i]["arch_name"].split("，")))
        name_j = list(map(str.strip, self.data[j]["arch_name"].split("，")))
        max_similarity = 0
        for i in range(len(name_i)):
            for j in range(len(name_j)):
                similarity = self.text_similarity(name_i[i], name_j[j])
                max_similarity = similarity if similarity > max_similarity else max_similarity
        return max_similarity

    def __arch_loc(self, i, j):
        weights= [0.1,0.3,0.6]
        loc_i =  list(map(str.strip, self.data[i]["arch_loc"].split("，")))
        loc_j = list(map(str.strip, self.data[j]["arch_loc"].split("，")))
        min_depth = min(len(loc_i), len(loc_j))

        similarity = 0
        for i in range(min_depth):
            if loc_i[i] == loc_j[i]:
                similarity += weights[i]
            else:
                return 0
        return similarity  

    def __parse_time(slef, time: str)  -> List[int]:
        result = []
        year = re.search(r"(\d)+年", time) 
        month = re.search(r"(\d)+月", time)
        if year:
            result.append(year.group())
        if month:
            result.append(month.group())     
        return result

    def __built_time(self, i, j):
        weights = [0.8, 0.2]
        time_i = self.data[i]["built_time"].strip()
        time_j = self.data[j]["built_time"].strip()
        lst_i = self.__parse_time(time_i)
        lst_j = self.__parse_time(time_j)
        min_depth = min(len(lst_i), len(lst_j))
        similarity = 0
        for i in range(min_depth):
            if lst_i[i].strip() == lst_j[i].strip():
                similarity += weights[i]  
            else:
                return 0
        return similarity

    def __get_img(self, image_url):
        response = requests.get(image_url)
        img = None
        if response.status_code == 200:
            image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv.imdecode(image_data, cv.IMREAD_COLOR)
        else:
            print(f"Failed to retrieve image: {image_url}. HTTP Status code: {response.status_code}")   

        #image check
        if img is not None:
            if img.shape[0] > 0 and img.shape[1] > 0:
                return img
        print(f"Failed to retrieve image: {image_url}.")
        return img

    def __get_imgs(self, i):
        urls = self.data[i]["img"]
        result = []
        for i in urls:
            img = self.__get_img( i)
            if img is not None:
                result.append(img)
        return result
    


    def __sift(self, img_1, img_2):
        sift = cv.SIFT.create()
        # 特征点提取与描述子生成
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)

        bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
        matches = bf.knnMatch(des1,des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        good_match_ratio = len(good_matches) / len(matches)
        if True:
            img_matches = cv.drawMatches(img_1, kp1, img_2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv.imshow("Good Matches", img_matches)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        
        print(f"Good Match Ratio: {good_match_ratio * 100:.2f}%")
        return good_match_ratio

    def __img(self, i, j):

        max_ratio = 0
        lst_i = self.__get_imgs(i)
        lst_j = self.__get_imgs(j)
        for i in  range(len(lst_i)):
            for j in range(len(lst_j)):
                ratio = self.__sift(lst_i[i], lst_j[j])
                max_ratio = ratio if ratio > max_ratio else max_ratio

        return max_ratio
        
    def total_similarity(self, i, j):
        weights = [1,1,1,1,1,1]
        similarity = 0
        feat_lst = list(Feat)
        for k in range(len(Feat)):
            value = self.similarity(i, j, feat_lst[k])
            similarity += weights[k] * value 
            print(f"{feat_lst[k].value} 相似度： {value}")
        return similarity / sum(weights) 
  

#integ = Integration(data_img) 
#print(integ.similarity(0,1, Feat.img))  



def sift(img_1, img_2):
    MIN_MATCH_COUNT = 10
    sift = cv.SIFT.create()
    # 特征点提取与描述子生成
    kp1, des1 = sift.detectAndCompute(img_1,None)
    kp2, des2 = sift.detectAndCompute(img_2,None)


    bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    matches = bf.knnMatch(des1,des2, k=2)

    # 逐步筛选并添加进度条
    good_matches = []
    for m, n in tqdm(matches,total=len(matches)):
    
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    good_match_ratio = len(good_matches) / min(len(kp1), len(kp2))
    if True:  
        height = max(img_1.shape[0], img_2.shape[0])
        width =  img_2.shape[1]  + img_2.shape[1] 
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        # 计算将img_1放置在左侧正中间的坐标位置
        img1_x = (width // 2 - img_1.shape[1] // 2)
        img1_y = (height - img_1.shape[0]) // 2

        # 将img_1放置到画布对应的位置上
        canvas[img1_y:img1_y + img_1.shape[0], img1_x:img1_x + img_1.shape[1]] = img_1
        
        img_matches = cv.drawMatches(img_1, kp1, img_2, kp2, good_matches[::2], canvas, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor = (0,255,0))
        
        cv.imwrite("matches_output.jpg", img_matches)
        cv.imshow("Good Matches", img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print(f"kp1: {len(kp1)}, kp2: {len(kp2)}, good match: {len(good_matches)}")
        print(f"Good Match Ratio: {good_match_ratio * 100:.2f}%")
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            cnt = 0
            for i in matchesMask:
                if i == 1:
                    cnt += 1
            print(f"matchesMask: {cnt}")        

            h,w,d = img_1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)

            img2 = cv.polylines(img_2,[np.int32(dst)],True,(0,0,255),1, cv.LINE_4)  
        else:
            print( "Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT) )
            matchesMask = None    
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask[::1], # draw only inliers
                        flags = 2)

        img3 = cv.drawMatches(img_1,kp1,img2,kp2,good_matches[::1],None,**draw_params)
        cv.imshow("Good Matches", img3)
        cv.waitKey(0)
        cv.destroyAllWindows()
    

    return good_match_ratio

img_1 = cv.imread("C:/Users/Wallie/Desktop/img_cmp/0033.jpg")
img_2 = cv.imread("C:/Users/Wallie/Desktop/img_cmp/008.jpeg")

def extract():
    sift = cv.SIFT.create()
    kp, des = sift.detectAndCompute(img_2,None)
    keypoints = np.array([kp[i].pt for i in range(len(kp))])
    db = DBSCAN(eps=50, min_samples=5).fit(keypoints)
    labels = db.labels_

    # 按照聚类结果，计算每个聚类的边界框
    for label in set(labels):
        if label == -1:  # 忽略噪声点
            continue
        cluster_points = keypoints[labels == label]
        x_min, y_min = np.min(cluster_points, axis=0)
        x_max, y_max = np.max(cluster_points, axis=0)
        
        # 绘制边界框
        img = cv.rectangle(img_2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv.imshow("Good Matches", img)
        cv.waitKey(0)
        cv.destroyAllWindows()    

def extract_1(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 使用GrabCut算法进行图像分割
    mask = np.zeros(gray.shape, np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)  # 定义一个矩形框来初始化前景和背景
    cv.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)

    # 将分割结果处理成主体区域
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_segmented = img * mask2[:, :, np.newaxis]

    # 获取主体区域的边界框
    contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow("Good Matches", img)
        cv.waitKey(0)
        cv.destroyAllWindows()           


sift(img_1, img_2)