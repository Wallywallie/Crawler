"""
第零步：构建一个合适的样本集：排列组合会遇到的多种情况
细粒度：标题、正文

案例 √ 
标题 √ 
正文 ×



第一步：暂且根据标题内容识别是否是同一个案例
潜在的方法：
1. 相似度分析
2. LSA
3. Named Entity Recognition, NER
4. 利用GPT进行识别

"""

import utills

def remove_suffix(strings):
    modified_strings = []
    for s in strings:
        # 检查字符串是否以"_relid"或"_added"结尾，并去除相应部分
        if s.endswith("_ArchDaily"):
            modified_strings.append(s[:-10])  # 去除"_relid"的7个字符
        if s.endswith("_ ArchDaily"):
            modified_strings.append(s[:-11])  # 去除"_relid"的7个字符            
        elif s.endswith("- 谷德设计网"):
            modified_strings.append(s[:-7])  # 去除"_added"的6个字符
        else:
            modified_strings.append(s)  # 保留原字符串
    return modified_strings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_pairs(strings, threshold=0.8):
    # 使用 CountVectorizer 将字符串转换为向量
    vectorizer = TfidfVectorizer().fit_transform(strings)
    vectors = vectorizer.toarray()

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)

    # 存储相似度超过阈值的配对
    similar_pairs = []
    num_strings = len(strings)

    for i in range(num_strings):
        for j in range(i + 1, num_strings):  # 避免重复比较
            print(strings[i] , "&", strings[j], ": ", cosine_sim[i][j])
            if cosine_sim[i][j] > threshold:
                similar_pairs.append((i, j))
                

    return similar_pairs

def test_1():
    """
    方法一：进行相似度分析
    1. 读取data下每一个文件的标题，两两比较相似度
    2. 利用并查集来记录相似对
    3. 打印结果

    实验结论：
    根据标题利用CountVectorizer判断相似度，表现不好，CountVectorizer会被关键词顺序影响
    利用TfidfVectorizer表现也不好难以区分地点不同但名称相同的建筑作品
    """    
    names = utills.get_title('data')
    names = remove_suffix(names)
    print(names)
    uf = utills.UnionFind(len(names))
    pr = get_similar_pairs(names, 0.8)


    for i in pr:
        uf.union(i[0], i[1])

    print(uf.get_groups())

def test_2():
    """
    方法二：
    使用LSA对文本进行主题分析

    实验结论：
    根据全文利用LSA（潜在语义分析）、LDA（潜在狄利克雷分布）进行主题分析，能够提取比如“空间”、“设计”之类的词语，但是无法提取案例的类别属性
    LSA Topics:
    (0, '0.628*"the" + 0.314*"and" + 0.265*"of" + 0.196*"Wen" + 0.196*"Studio" + 0.157*"空间" + 0.147*"设计" + 0.118*"The" + 0.118*"to" + 0.098*"芝作室"')
    """
    from gensim.models import LsiModel, LdaModel
    from gensim.corpora import Dictionary
    import jieba

    #读取一个文本的内容
    path = 'D:\CS\crawler\data\北京尼康Nikon直营店 _ 芝作室  - 谷德设计网.txt'
    txt = []
    with open(path, 'r', encoding = 'utf-8') as f:
        txt.append(f.read())
        
    # 停用词列表（根据需要自定义）
    stopwords = set()
    stopwords.add("。")
    """
    with open('D:/CS/crawler/data/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords.update([line.strip() for line in f.readlines()])
    """

    # 中文文本分词，并去掉停用词
    def preprocess_text(text):
        # 使用jieba进行分词
        tokens = jieba.cut(text)
        # 去掉停用词，并过滤掉单个字符的词
        return [token for token in tokens if token not in stopwords and len(token) > 1]


    tokenized_texts = [preprocess_text(text) for text in txt]

    # 建立词典和语料库
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # 使用LSA模型进行主题建模和主题分析
    lsa_model = LsiModel(corpus, num_topics=2, id2word=dictionary)
    lsa_topics = lsa_model.print_topics()

    # 使用LDA模型进行主题建模和主题分析
    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)
    lda_topics = lda_model.print_topics()

    # 打印LSA主题
    print("LSA Topics:")
    for topic in lsa_topics:
        print(topic)

    # 打印LDA主题
    print("\nLDA Topics:")
    for topic in lda_topics:
        print(topic)

def test_3():

    """
    方法三：Named Entity Recognition, NER
    1. 提取标题中的建筑名称？？能否根据全文提取？？
    2. 归并？？根据建筑师、建造时间、建造地址
    3. 返回json数据展示在前端

    实验结论：使用预训练bert模型直接处理效果不好，不finetune的话，没有办法直接识别标题中的建筑名称
    [{'entity': 'LABEL_0', 'score': 0.6752568, 'index': 1, 'word': 'the', 'start': 0, 'end': 3}, {'entity': 'LABEL_1', 'score': 0.5435043, 'index': 2, 'word': 'shanghai', 'start': 4, 'end': 12}, {'entity': 'LABEL_0', 'score': 0.6461245, 'index': 3, 'word': 'tower', 'start': 13, 'end': 18}, {'entity': 'LABEL_0', 'score': 0.6208189, 'index': 4, 'word': 'is', 'start': 19, 'end': 21}, {'entity': 'LABEL_0', 'score': 0.7275482, 'index': 5, 'word': 'a', 'start': 22, 'end': 23}, {'entity': 'LABEL_0', 'score': 0.56183803, 'index': 6, 'word': '63', 'start': 24, 'end': 26}, {'entity': 'LABEL_0', 'score': 0.707379, 'index': 7, 'word': '##2', 'start': 26, 'end': 27}, {'entity': 'LABEL_0', 'score': 0.7050673, 'index': 8, 'word': '-', 'start': 27, 'end': 28}, {'entity': 'LABEL_0', 'score': 0.59342724, 'index': 9, 'word': 'meter', 'start': 28, 'end': 33}, {'entity': 'LABEL_0', 'score': 0.6844956, 'index': 10, 'word': '-', 'start': 33, 'end': 34}, {'entity': 'LABEL_0', 'score': 0.58525175, 'index': 11, 'word': 'tall', 'start': 34, 'end': 38}, {'entity': 'LABEL_0', 'score': 0.6316937, 'index': 12, 'word': 'skyscraper', 'start': 39, 'end': 49}, {'entity': 'LABEL_0', 'score': 0.57929444, 'index': 13, 'word': 'located', 'start': 50, 'end': 57}, {'entity': 'LABEL_0', 'score': 0.61941737, 'index': 14, 'word': 'in', 'start': 58, 'end': 60}, {'entity': 'LABEL_0', 'score': 0.55807936, 'index': 15, 'word': 'lu', 'start': 61, 'end': 63}, {'entity': 'LABEL_0', 'score': 0.62735593, 'index': 16, 'word': '##jia', 'start': 63, 'end': 66}, {'entity': 'LABEL_0', 'score': 0.6028534, 'index': 17, 'word': '##zu', 'start': 66, 'end': 68}, {'entity': 'LABEL_0', 'score': 0.5975958, 'index': 18, 'word': '##i', 'start': 68, 'end': 69}, {'entity': 'LABEL_0', 'score': 0.57532406, 'index': 19, 'word': '.', 'start': 69, 'end': 70}]

    下一步:
    - 考虑方案四:初步结论-在不微调大模型的情况下，大模型的正确率与微调后的bert相似，但大模型的输出格式难以固定，
    - finetune: 
        实验步骤：
            1. 以公共建筑-居住建筑-工业建筑三大主题关键词爬取约700条数据，对数据进行标注
                其中bad case类别：
                    标题            建筑名称          描述对象       是否包含
                    不同标题 抽取出 相同的建筑名称 -> 同一个描述对象      √
                    不同标题 抽取出 相同的建筑名称 -> 不同描述对象        
                    不同标题 抽取出 不同建筑名称 -> 同一个描述对象
                    不同标题 抽取出 不同建筑名称 -> 不同描述对象          √
                    相同标题 抽取出 相同的建筑名称 -> 不同描述对象
                    无法从标题中抽取建筑名称      -> 不包含建筑名称       √
                    无法从标题中抽取建筑名称      -> 不是建筑案例         √

            2. 运用keras框架，微调预处理bert模型，训练20个epoch,accurancy达到78%

    下一步：
        1.提升accurancy到90% ：继续提高epoch; 增加有效数据的数量  
        2.构造包含各种bad case的测试集

    下下一步：
        1.跑通标题的提取之后，试着拓展到全文，从全文提取多个特征：
        [建筑师/事务所] [地点] [时间] [材料]
    """

    from transformers import pipeline
    import tqdm

    # 使用预训练的BERT模型进行NER
    model_path = "D://CS//crawler//bert//chinese_L-12_H-768_A-12"
    ner_model = pipeline('ner', model="bert-base-uncased")
    text = "The Shanghai Tower is a 632-meter-tall skyscraper located in Lujiazui."
    entities = ner_model(text)
    print(entities)



test_3()


