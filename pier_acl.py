import re
import json
import essay
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.externals import joblib
from extract_feature import  Features

def cleanText(string):
    pattern_space = r'\s{2,}'
    pattern_juhao = r'\.'
    pattern_gantanhao = r'\!'
    pattern_wenhao = r'\!'
    pattern_douhao = r'\,'

    string = re.sub(pattern_space,' ',string)
    string = re.sub(pattern_douhao, ', ', string)
    string = re.sub(pattern_juhao, '. ', string)
    string = re.sub(pattern_gantanhao, '! ', string)
    string = re.sub(pattern_wenhao,'? ', string)
    return string

def parsing_train_set_and_chanfen(file_name,output_file_x,output_file_y):

    with open(file_name,'r',errors='ignore') as f:
        with open(output_file_x,'w',errors='ignore') as f_x :
            line = f.readline()
            with open(output_file_y,'w',errors='ignore') as f_y:
                while line:
                    list_essay_info = line.split('\t')
                    essay_extracted = (json.loads(json.loads(list_essay_info[0]))).strip() # 文章
                    essay_extracted = cleanText(essay_extracted)
                    f_x.write(essay_extracted + '\n')  # 写入只含文章的文件
                    f_y.write(list_essay_info[1] + '\n')
                    line = f.readline()
                # if essay.isspace == True:
                #     print("space_essay is here")
                # essay = cleanText(essay)
                # scores_views_point = list[2]  # 每篇文章的各个角度的分数    ---->词汇 句子 篇章结构 内容相关
                # table_total = list[1]  # 文章的总打分

def parse_token_pos_lemma(essay_object,lemmatizer):
    """
    通过nltk的语料库训练pos模型，然后拿文章进行token，然后得pos,增加lemma
    :param essay_object: 
    :return: 返回的是一篇文章的tokens和token对应的pos，lemma
    """
    VerbTags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    NounTags = ['NN', 'NNS', 'NNP', 'NNPS', 'Unk']
    AdjectiveTags = ['JJ', 'JJR', 'JJS']
    AdverbTags = ['RB', 'RBR', 'RBS']

    essay_token_attribute = nltk.pos_tag(essay_object.tokens)  # [tuple(token,pos)......]
    essay_token_attributeed = []
    for token_attribute in essay_token_attribute:
        token_attribute = list(token_attribute)
        if token_attribute[1] in VerbTags:
            # print("v")
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='v'))
            # print(token_attribute[2])
        elif (token_attribute[1] in NounTags) or (token_attribute[1] == 'Unk'):
            # print("n")
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='n'))
        elif token_attribute[1] in AdjectiveTags:
            # print("a")
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='a'))
        elif token_attribute[1] in AdverbTags:
            # print("r")
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='r'))
        else:
            token_attribute.append(token_attribute[0])
        essay_token_attributeed.append(token_attribute)
    # print(essay_token_attributeed)
    return essay_token_attributeed

def return_essay_object(file_name):
    """
    获取训练集所有文章内容，按照对象形式存储
    :param file_name: 源训练集
    :return: 对象列表
    """
    lemmatizer = WordNetLemmatizer()  # 抽词元(词形还原)  2
    essays_object_list = []
    with open(file_name, 'r', errors='ignore') as f:
        line = f.readline()
        i=0
        with open(r'./rdata/object_essay_wanzheng10.pkl','wb') as f_pkl_object:
            while line:
                paragraphs = []
                list_essay_info = line.split('\t')
                essay_extracted = (json.loads(json.loads(list_essay_info[0]))) # 文章
                essay_extracted = cleanText(essay_extracted)  # 清洗
                # num = 1
                for paragraph in essay_extracted.split('\n'): # 存储
                    paragraphed = paragraph.strip()
                    # if not paragraphed.isspace():    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 警告这里还没有处理完，无法判断内容为空字符却添加的异常
                        # print((paragraph.strip()))
                        # print(num)
                        # num += 1
                    paragraphs.append(paragraphed)
                # print("上面是一片文章的段落区分")
                essay_object = essay.Essay()
                essay_object.essay_str = essay_extracted  # 清洗后的文章
                essay_object.paragraphs = paragraphs  # 文章的段落列表
                essay_object.sentences = nltk.sent_tokenize(essay_extracted) # 文章的句子列表
                essay_object.tokens = nltk.word_tokenize(essay_extracted)  # 文章的token列表
                essay_object.tokens_pos_lemma = parse_token_pos_lemma(essay_object,lemmatizer)  # 文章token,pos,lemma  注释：[0,1,2]-->[token,pos,lemma]
                essay_object.paragraphs_num = len(paragraphs)  # 文章的段落数
                essay_object.essay_words_num = len(nltk.word_tokenize(essay_extracted))  # 文章的长度

                pickle.dump(essay_object,f_pkl_object)
                # essays_object_list.append(essay_object)
                i += 1
                print("文章对象--{}---写入完成".format(i))
                # if i == 10000:
                #     break
                line = f.readline()
    # return essays_object_list

def extractor_Y_train():
    """
    解析训练集的数据
    :return: 
    """
    Y_train = []
    # i = 0
    with open(r'./train_test_set/clean_essays_y.train','r',errors='ignore') as f:
        line = f.readline()
        while line:
            # print(line)
            line = float(line.strip())
            Y_train.append(line)
            # i += 1
            # if i == 10000:
            #     break
            line = f.readline()
    return  np.array(Y_train)

def parsing_web_data(essay_text):
    essay_text_cleaned = cleanText(essay_text)   # 清洗数据
    essay_object = essay.Essay()  # 创建essay对象
    f = Features()  # 创建特征对象
    lemmatizer = WordNetLemmatizer()  # 抽词元(词形还原)  2

    paragraphs = []
    for paragraph in essay_text_cleaned.split('\n'):  # 处理段落
        paragraphed = paragraph.strip()
        paragraphs.append(paragraphed)

    essay_object.essay_str = essay_text_cleaned  # 清洗后的文章
    essay_object.paragraphs = paragraphs  # 文章的段落列表
    essay_object.sentences = nltk.sent_tokenize(essay_text_cleaned)  # 文章的句子列表
    essay_object.tokens = nltk.word_tokenize(essay_text_cleaned)  # 文章的token列表
    essay_object.tokens_pos_lemma = parse_token_pos_lemma(essay_object,lemmatizer)  # 文章token,pos,lemma  注释：[0,1,2]-->[token,pos,lemma]
    essay_object.paragraphs_num = len(paragraphs)  # 文章的段落数
    essay_object.essay_words_num = len(nltk.word_tokenize(essay_text_cleaned))  # 文章的长度

    return f.returnFeatures(essay_object)


if  __name__ == "__main__":
    # return_essay_object(r'./train_test_set/pigai.train')
    score_list = []
    model = joblib.load(r'./model/basic_liner_model_allData2.pkl')  # 模型加载
    essay_object = essay.Essay()  # 创建essay对象
    fs = Features()  # 创建特征对象
    lemmatizer = WordNetLemmatizer()  # 抽词元(词形还原)  2

    with open(r'./train_test_set/pigaiessay.test', 'r', errors='ignore') as f:  # 测试数据加载
        with open(r'./rdata/test_result_2_all.txt', 'w', errors='ignore') as ff:  # 结果数据写入
            i = 1
            line = f.readline()
            while line:
                paragraphs = []
                essay_text = json.loads(line)
                essay_extracted = cleanText(essay_text)  # 清洗

                for paragraph in essay_extracted.split('\n'):
                    paragraphed = paragraph.strip()
                    paragraphs.append(paragraphed)

                essay_object = essay.Essay()
                essay_object.essay_str = essay_extracted  # 清洗后的文章
                essay_object.paragraphs = paragraphs  # 文章的段落列表
                essay_object.sentences = nltk.sent_tokenize(essay_extracted)  # 文章的句子列表
                essay_object.tokens = nltk.word_tokenize(essay_extracted)  # 文章的token列表
                essay_object.tokens_pos_lemma = parse_token_pos_lemma(essay_object,lemmatizer)  # 文章token,pos,lemma  注释：[0,1,2]-->[token,pos,lemma]
                essay_object.paragraphs_num = len(paragraphs)  # 文章的段落数
                essay_object.essay_words_num = len(nltk.word_tokenize(essay_extracted))  # 文章的长度

                features = np.array(fs.returnFeatures(essay_object)).reshape(1,-1)
                # print(features)
                # poly = PolynomialFeatures(degree=2, include_bias=False)
                # X_train_poly = poly.fit_transform(features)
                # score = model.predict(X_train_poly)
                score = model.predict(features)
                score = score.tolist()
                score = ''.join(str(s) for s in score)
                print(score)
                ff.write(str(i)+' '+ score + '\n')
                # score_list.append(score)
                # print(score)
                # print(i)
                i += 1
                line = f.readline()

    # with open(r'./rdata/test_result.txt','w',errors='ignore') as ff:
    #
        # s_list = s.tolist()
        # str = ''.join(str(s) for s in s_list)
