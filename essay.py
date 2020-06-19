
class Paragraph:
    sentences = [] # 一段的句子列表
    paragraph_words_num = 0  # 一个段落内的字数


class Essay:
    essay_str = ""  # 文章
    paragraphs = []  # 一篇文章的段落列表
    sentences = []  # 一篇文章的句子列表
    tokens = []  # 文章的token列表
    essay_words_num = 0 # 一篇文章长度
    essay_words_level_num = 0 # 一篇文章大于单词level大于school_level的字数
    paragraphs_num = 0 # 一篇文章的段落数
    tokens_pos_lemma = []  # 文章的token pos,lemma   [[token1,pos,lemma]...[token2,pos,lemma].]

class Sentence:
    sentence = ""
    sentence_num = 0 # 句子的长度
    sentence_grammar_error = True # 句子是否有语法错误

class token:
    token = ""
    token_spell_error = True # 单词是否有拼写错误
    pos = "" # 单词的词性
    level = 0 # 单词的level

    def __init__(self,word):
        self.word = word

    def spellcheck(self):
        word = self.word
        pass
    def word_level(self,word):
        word = self.word
        pass