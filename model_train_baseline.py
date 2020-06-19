from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
from parsingText import extractor_Y_train
from  extract_feature import *
import pickle
import numpy as np

# 数据的加载
X_train = []
fea = Features()
with open(r'./rdata/object_essay_allData_39421.pkl','rb') as f:
    for i in range(39421):
        s = fea.returnFeatures(pickle.load(f))
        X_train.append(s)

Y_train = extractor_Y_train(39421)  # done
X_train = np.array(X_train)  # done

# 训练线性回归模型
model_1 = linear_model.LinearRegression()
model_1.fit(X_train,Y_train)

# 训练多项式的线性回归模型
# poly = PolynomialFeatures(degree=2,include_bias=False)
# X_train_poly = poly.fit_transform(X_train)
# model_2 = linear_model.LinearRegression(normalize=True)
# model_2.fit(X_train_poly,Y_train)

# 模型保存
joblib.dump(model_1,'./model/basic_liner_model_allData_add_grammar.pkl')
# joblib.dump(model_2,'./model/poly_liner_model_10.pkl')
