# import enchant
#import grammar_check
from math import exp
import nltk
import numpy as np
from spellchecker import SpellChecker
from os.path import join,dirname,realpath,abspath
import requests
import json


class Features:
    ave_words_length = 0.0  # 文章的平均单词长度                    done
    words_length_variance = 0.0  # 文章的单词长度方差               done
    cet4_ratio = 0.0  # 四级词汇使用比例                            done
    cet6_ratio = 0.0  # 六级词汇使用比例                            done
    Preposition_usage_metric = 0.0  # 介词使用度量                  done
    definite_article_usage_metric = 0.0  # 定冠词使用度量           done
    spell_error_word_ratio = 0.0  # 单词拼写错误占比                done

    punctuation_usage_metric = 0.0  # 标点符号使用度量
    sentences_grammar_error_ratio = 0.0  # 文章内句子语法错误数占比
    ave_sentences_length = 0.0 # 文章内句子的平均长度               done
    sentences_length_variance = 0.0  # 文章的句子长度方差           done
    SentenceReadability = 0.0  # 句子的可读性                       done
    Appropriate_number_of_sentences = 0.0  # 句子的数量使用度量     done
    connect_word_ratio = 0.0  # 连接词占所有词的比例                done

    paragraphs_word_ratio = 0.0  # 段落内字数占比
    paragraphs_num = 0  # 段落数                                    done

    content_describe = 0.0  # 文章内容描述度量  段落描述的相似性
    word_sentence_covariance = 0.0  # 单词句子的协方差
    sentence_paragraph_covariance = 0.0  #  句子段落的协方差
    sentence_length = 0  # 文章长度                                 done
    word_and_sentence_length_ratio = 0.0  # 单词数量与句子数量的比值 done

    def returnFeatures(self,essay_object):
        conn_list = create_connwords_list(join(dirname(realpath(__file__)),'./rdata/connwords.txt'))
        features = []
        self.ave_words_length = cal_ave_words_length_essay(essay_object)
        self.words_length_variance = cal_words_length_variance(essay_object)
        self.cet4_ratio = float(1 if (len([value for value in levelToken(essay_object) if value == 2])) == 0 else len([value for value in levelToken(essay_object) if value == 2])) / essay_object.essay_words_num
        self.cet6_ratio = float(1 if (len([value for value in levelToken(essay_object) if value == 3])) == 0 else len([value for value in levelToken(essay_object) if value == 3])) / essay_object.essay_words_num
        self.Preposition_usage_metric = abs((float(len([pos for pos in (token_pos_lemma[1] for token_pos_lemma in essay_object.tokens_pos_lemma) if pos =='IN'])) / essay_object.essay_words_num) - 0.0321)
        self.definite_article_usage_metric = abs((float(len([ding for ding in (token_pos_lemma[1] for token_pos_lemma in essay_object.tokens_pos_lemma) if ding =='the'])) / essay_object.essay_words_num) - 0.065)
        self.spell_error_word_ratio = exp(- float(spell_check_spellchecker(essay_object)) /essay_object.essay_words_num)
        self.ave_sentences_length = cal_ave_senteces_length_essay(essay_object)
        self.sentences_length_variance = cal_variance_senteces_length_essay(essay_object)
        self.SentenceReadability = 0.47 * self.ave_words_length + 0.5 * self.ave_sentences_length - 21.43
        self.Appropriate_number_of_sentences = float(exp(13 - len(essay_object.sentences)) if len(essay_object.sentences)>= 13 else exp(len(essay_object.sentences) - 13))    # 偶函数
        self.connect_word_ratio = float(len([word for word in essay_object.tokens if word in  conn_list])) / essay_object.essay_words_num
        self.paragraphs_num = exp(3 - essay_object.paragraphs_num) if essay_object.paragraphs_num >= 3 else exp(essay_object.paragraphs_num - 3)   # 人为度量
        self.sentence_length = exp(155 - essay_object.paragraphs_num) if essay_object.paragraphs_num >= 155 else exp(essay_object.paragraphs_num - 155)  # 人为度量
        self.sentences_grammar_error_ratio = float(grammar_check_local(essay_object)) / len(essay_object.sentences)
        self.word_and_sentence_length_ratio = float(essay_object.essay_words_num) / len(essay_object.sentences)
        features.append(self.ave_words_length)
        features.append(self.words_length_variance)
        features.append(self.cet4_ratio)
        features.append(self.cet6_ratio)
        features.append(self.Preposition_usage_metric)
        features.append(self.definite_article_usage_metric)
        features.append(self.spell_error_word_ratio)
        features.append(self.ave_sentences_length)
        features.append(self.sentences_length_variance)
        features.append(self.SentenceReadability)
        features.append(self.Appropriate_number_of_sentences)
        features.append(self.connect_word_ratio)
        features.append(self.paragraphs_num)
        features.append(self.sentence_length)
        features.append(self.sentences_grammar_error_ratio)
        features.append(self.word_and_sentence_length_ratio)
        # print(features)
        return features




def Break_up_essay(essay_str_atribute):
    """
    拆分文章
    :param essay: 
    :return: 返回段落列表
    """
    paragraphs_list = []
    for paragraph in essay_str_atribute.split('\n'):
        paragraph = paragraph.strip()
        paragraphs_list.append(paragraph)
    return paragraphs_list

def Break_up_paragraph(paragraph):
    """
    拆分段落为句子
    :param paragraph: 
    :return: 句子列表
    """
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def Break_up_sentece(sentence):
    """
    把句子切分成token
    :param sentence: 
    :return: token列表
    """
    tokens_list = nltk.word_tokenize(sentence)
    return tokens_list

def cal_words_length_variance(essay_object):
    """
    文章的单词长度方差
    :param essay_object: 
    :return: 
    """
    list_word_len_essay = []
    for word in nltk.word_tokenize(essay_object.essay_str):
        list_word_len_essay.append(len(word))
    return np.var(list_word_len_essay)

def cal_essay_length(essay_para):
    """
    计算文章长度
    :param essay_para: 
    :return: 
    """
    print(len(nltk.word_tokenize(essay_para)))

def cal_ave_words_length_essay(essay_object):
    """
    计算单词的平均长度
    :param essay_object: 
    :return: 
    """
    list_word_len_essay = []
    for word in nltk.word_tokenize(essay_object.essay_str):
        list_word_len_essay.append(len(word))
    return np.mean(list_word_len_essay)

def cal_ave_senteces_length_essay(essay_object):
    """
    计算文章的句子平均长度
    :param essay_object: 
    :return: 
    """
    list_sentence_len_essay = []
    len_sentence = 0
    for sentence in essay_object.sentences:
        len_sentence = len(nltk.word_tokenize(sentence))
        list_sentence_len_essay.append(len_sentence)
    return np.mean(list_sentence_len_essay)

def cal_variance_senteces_length_essay(essay_object):
    """
    计算文章的句子长度方差
    :param essay_object: 
    :return: 
    """
    list_sentence_len_essay = []
    len_sentence = 0
    for sentence in essay_object.sentences:
        len_sentence = len(nltk.word_tokenize(sentence))
        list_sentence_len_essay.append(len_sentence)
    return np.var(list_sentence_len_essay)

def cal_length_sentence(sentence):
    """
    计算句子的长度，包含标点
    :param sentence: 
    :return: 
    """
    return len(nltk.word_tokenize(sentence))

def cal_length_word(word):
    """
    计算单词的长度
    :param word: 
    :return: 
    """
    return len(word)

def create_CET4_list(path):
    # print("3")
    cet4_words = []
    with open(path,'r',errors='ignore') as f:
        for line in f.readlines():
            cet4_words.append(line.split()[0])
    return  cet4_words

def create_CET6_list(path):
    # print("2")
    cet6_words = []
    with open(path,'r',errors='ignore') as f:
        for line in f.readlines():
            cet6_words.append(line.split()[0])
    return  cet6_words

def create_school_list(path):
    # print("1")
    school_words = []
    print(path)
    with open(path,'r',errors='ignore') as f:
        for line in f.readlines():
            school_words.append(line.split()[0])
    return  school_words

EnglishPunct = ['.', ',', '?', '!', ';', ':', '%', '"', '\'', '-']
SpecialWords = ["'s", "'m", "'re", "ca", "n't", "'d", "'ll", "'ve", "isn", "hasn", "havn", "wouldn", "shouldn",
                "wasn", "aren", "doesn", "weren", "won", "needn"]
NumberTags = ['CD']

SchoolWords = create_school_list(join(dirname(realpath(__file__)),r'rdata/school.dic'))
CET4Words = create_CET4_list(join(dirname(realpath(__file__)),r'rdata/college4.dic'))
CET6Words = create_CET6_list(join(dirname(realpath(__file__)),r'rdata/college6.dic'))

def create_connwords_list(path):
    """
    衔接词表
    :param path: 
    :return: 
    """
    connwords = []
    with open(path,'r',errors='ignore') as f:
        for line in f.readlines():
            connwords.append(line.strip())
    connwordsSet = set([x for x in connwords if len(x.split()) > 0])
    return connwordsSet

def create_stopwords_list():
    return  set(nltk.corpus.stopwords.words("english"))

def parse_token_pos(essay_object):
    """
    通过nltk的语料库训练pos模型，然后拿文章进行token，然后得pos
    :param essay_object: 
    :return: 返回的是一篇文章的tokens和token对应的pos
    """
    # train pos by nltk's cropus
    from nltk.corpus import treebank
    train_sents = treebank.tagged_sents()[:3000]
    test_sents = treebank.tagged_sents()[3000:]

    train_brown = nltk.corpus.brown.tagged_sents()[0:5000]
    test_brown = nltk.corpus.brown.tagged_sents()[5000:]

    tnt_tagger = nltk.tag.tnt.TnT()
    tnt_tagger.train(train_sents)

    t_tagger_brown = nltk.tag.tnt.TnT()
    t_tagger_brown.train(train_brown)

    print("训练pos模型完成")
    print("当前文章为{}".format(essay_object.essay_str))
    tokenTags = tnt_tagger.tag(essay_object.tokens)  # pos of token
    bTags = t_tagger_brown.tag(essay_object.tokens)  # pos of token
    essay_token_attribute = []
    for tuple_token_pos in tokenTags:  # change token
        list_token_pos = list(tuple_token_pos)
        if list_token_pos[1] == 'Unk':
            list_token_pos[1] = bTags[0][1]
        if list_token_pos[1] == 'Unk':
            if list_token_pos[0][-2:] == 'ed':
                list_token_pos[1] = 'VBD'
        essay_token_attribute.append(list_token_pos)

    return essay_token_attribute

def lemma_create(essay_tokens_and_poss):
    """
    为token_attribute_list增加lemma属性
    :param essay_tokens_shuxing_list: 
    :return: essay_tokens_shuxing_list ----->[tokens[token,pos,lemma]]
    """
    VerbTags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ProperNounTags = ['NNP', 'NNPS']
    NounTags = ['NN', 'NNS', 'NNP', 'NNPS', 'Unk']
    AdjectiveTags = ['JJ', 'JJR', 'JJS']
    AdverbTags = ['RB', 'RBR', 'RBS']
    PronounTags = ['PRP', 'PRP$', 'WP', 'WP$']
    RestPronounTags = ['WP', 'WP$']
    NumberTags = ['CD']

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()  # 抽词元(词形还原)  2
    for token_attribute in essay_tokens_and_poss:
        if token_attribute[1] in VerbTags:
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='v'))
        elif (token_attribute[1] in NounTags) or (token_attribute[1] == 'Unk'):
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='n'))
        elif token_attribute[1] in AdjectiveTags:
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='a'))
        elif token_attribute[1] in AdverbTags:
            token_attribute.append(lemmatizer.lemmatize(token_attribute[0], pos='r'))
        else:
            token_attribute.append(token_attribute[0])
    return essay_tokens_and_poss

def levelToken(essay_object):
    """
    拿到一篇文章的token的level list
    :param essay_tokens_shuxing_list: 一篇文章的tokens   tokens[token,pos,lemma] 
    :return: 返回一篇文章的level 列表，用来做词级占比
    
    """
    essay_list_level = []
    for token in essay_object.tokens_pos_lemma:
        w = token[0]
        lw = token[0].lower()
        lemma = token[2]
        if w in EnglishPunct or token[1] in NumberTags:
            essay_list_level.append(0)
        elif (w in SchoolWords) or (lw in SchoolWords) or (lemma in SchoolWords) or (w in SpecialWords):
            essay_list_level.append(1)
        elif (w in CET4Words) or (lw in CET4Words) or (lemma in CET4Words):
            essay_list_level.append(2)
        elif (w in CET6Words) or (lw in CET6Words) or (lemma in CET6Words):
            essay_list_level.append(3)
        else:
            essay_list_level.append(4)

    # print(essay_list_level)
    return essay_list_level

# def specll_check_pyenchant(essay_object):
#     """
#     查询一篇文章的拼写错误，返回拼写错误的错误个数
#     :param essay_object:
#     :return:
#     """
#     spell_error_num = 0
#     d = enchant.Dict("en_US")
#     for token in essay_object.tokens:
#         if d.check(token) == False:
#             spell_error_num  += 1
#     return spell_error_num

def spell_check_spellchecker(essay_object):
    spell = SpellChecker()
    misspelled = spell.unknown(essay_object.tokens)
    return len(misspelled)

def grammar_check_local(essay_object):
    """
    记录一篇文章内语法错误数
    :param essay_object: 
    :return: 
    """
    url = 'http://localhost:8078/v2/check'
    headers = {'content-type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
    payload = {'text':'{}'.format(" ".join(essay_object.sentences)),'language':'en-US'}
    ret = requests.post(url, data=payload, headers=headers)
    data = json.loads(ret.text)
    matches = data['matches']
    return len(matches)



if __name__ == "__main__":
    pass






