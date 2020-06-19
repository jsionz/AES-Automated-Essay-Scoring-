import xlrd
import numpy as np
from parsingText import parsing_web_data
from sklearn.externals import joblib
from os.path import join,dirname,realpath

model_1 = joblib.load(join(dirname(realpath(__file__)),'./model/basic_liner_model_allData_add_grammar.pkl'))
data = xlrd.open_workbook(r'cet4_test.xlsx')
sheet = data.sheets()[0]
# print(sheet)

essays_str = sheet.col_values(2)
num = 0
with open(r'cet4_test_reuslt_add_grammar_feature.txt','w',errors='ignore') as f:
    for essay_str in essays_str:
        features = parsing_web_data(essay_str)
        features = np.array(features).reshape(1, -1)
        result = model_1.predict(features)
        num += 1
        score = result[0]
        print(num)
        f.write(str(score) + '\n')
