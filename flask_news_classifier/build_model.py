from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from .setting_params import *
from tensorflow.keras.preprocessing import sequence
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from .utils import *
import json
import jieba
import numpy as np
import requests
import tensorflow as tf
import os
import shutil




class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=5,
                 last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


class NewsDector(object):
    def __init__(self):
        if not os.path.exists('./cnn_model.h5'):
            print('开始训练模型')
            self.train_model()
        self.model = self.load_model()

    def train_model(self): # 训练模型
        # 如果不存在词汇表，重建
        if not os.path.exists(vocab_file):
            build_vocab(data_dir, vocab_file, vocab_size)
        # 获得 词汇/类别 与id映射字典
        categories, cat_to_id = read_category()
        words, word_to_id = read_vocab(vocab_file)

        # 全部数据
        x, y = read_files(data_dir)
        data = list(zip(x,y))
        del x,y
        # 乱序
        random.shuffle(data)
        # 切分训练集和测试集
        train_data, test_data = train_test_split(data)
        # 对文本的词id和类别id进行编码
        x_train = encode_sentences([content[0] for content in train_data], word_to_id)
        y_train = to_categorical(encode_cate([content[1] for content in train_data], cat_to_id))
        x_test = encode_sentences([content[0] for content in test_data], word_to_id)
        y_test = to_categorical(encode_cate([content[1] for content in test_data], cat_to_id))

        print('对序列做padding，保证是 samples*timestep 的维度')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        print('构建模型...')
        model = TextCNN(maxlen, max_features, embedding_dims).get_model()
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        print('训练...')

        # 设定callbacks回调函数
        my_callbacks = [
            ModelCheckpoint('./cnn_model.h5', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
        ]

        # fit拟合数据
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=my_callbacks,
                  validation_data=(x_test, y_test))
        model.save('./cnn_model.h5')


    def load_model(self): #加载模型
        import os
        model = tf.keras.models.load_model('./cnn_model.h5')
        return model

    def save_load_model_tf_serving(self): # tensorflow serving 运行测试

        # 指定路径
        if os.path.exists('./Models/CNN/1'):
            shutil.rmtree('./Models/CNN/1')

        export_path = './Models/CNN/1'
        # 导出tensorflow模型以便部署
        model = self.load_model()
        tf.saved_model.save(model, export_path)
        MODEL_DIR = os.getcwd() + "/Models/CNN"
        os.environ["MODEL_DIR"] = MODEL_DIR
        # os.system("nohup tensorflow_model_server --rest_api_port=8501 --model_base_path='${MODEL_DIR}'")
        os.system("docker run -p 8501:8501 --mount type=bind,source=/Users/flybird/Desktop/YRUN/URun.ResearchPrototype/People/Xiaoxian/新闻多分类/News-Classifier-Machine-Learning-and-Deep-Learning/Models/CNN,target=/models/cnn_serving -e MODEL_NAME=cnn_serving -t tensorflow/serving &")


    def process_test_data(self,X): # 处理预测文本数据
        if not os.path.exists(vocab_file):
            build_vocab(data_dir, vocab_file, vocab_size)
        # 获得 词汇/类别 与id映射字典
        categories, cat_to_id = read_category()
        words, word_to_id = read_vocab(vocab_file)
            # 获得 词汇/类别 与id映射字典
        words, word_to_id = read_vocab(vocab_file)
        text = X
        print(jieba.lcut(text))
        text_seg = encode_sentences([jieba.lcut(text)], word_to_id)
        text_input = sequence.pad_sequences(text_seg, maxlen=maxlen)
        return text_input

    def predict(self,X):
        X = self.process_test_data(X)
        proba = self.model.predict(X)
        news_dict = {'0': '汽车', '1': '娱乐', '2': '军事', '3': '体育', '4': '科技'}
        print('文章属于:{}类别'.format(news_dict[str(np.argmax(proba))]))
        print(proba)
        print(str(np.argmax(proba)))
        return str(np.argmax(proba))


    def tf_run(self,text): # 运行测试
        text_input = self.process_test_data(text)
        data = json.dumps({"signature_name": "serving_default",
                           "instances": text_input.reshape(1,100).tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post('http://localhost:8501/v1/models/cnn_serving:predict',
                                      data=data, headers=headers)
        proba = json_response.text.split(':')[1].strip()[2:-9].split(',')
        proba = [float(i) for i in proba]

        news_dict = {'0': '汽车', '1': '娱乐', '2': '军事', '3': '体育', '4': '科技'}
        print('文章属于:{}类别'.format(news_dict[str(np.argmax(proba))]))
        print(proba)
        print(str(np.argmax(proba)))
        return str(np.argmax(proba))


if __name__ == '__main__':
    # model = tf.keras.models.load_model('./cnn_model.h5')
    try:
        news = NewsDector()
        # if not os.path.exists('./cnn_model.h5'):
        #     news.train_model()
        news.predict('雷克萨斯汽车时速可以达到200/km')
    except Exception as e:
        print(e)