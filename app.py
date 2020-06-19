from flask import Flask, render_template, request
import numpy as np
from parsingText import parsing_web_data
from sklearn.externals import joblib
from os.path import join,dirname,realpath

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/postchuli', methods=['GET', 'POST'])
def chuli():
    if request.method == 'GET':
        return '<h1>get</h1>'
    elif request.method == 'POST':
        essay = request.form.get('essay')  # get essay
        print("请求接收完成")
        features = parsing_web_data(essay)
        features = np.array(features).reshape(1,-1)
        print("数据处理完成")
        # 加载模型
        print("模型加载中")
        model_1 = joblib.load(join(dirname(realpath(__file__)),'./model/basic_liner_model_allData2.pkl'))
        # 预测结果
        result = model_1.predict(features)
        print("结果返回")
        return '<h1>post{}</h1>'.format(result)


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=8051)
