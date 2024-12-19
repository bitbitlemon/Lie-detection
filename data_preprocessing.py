# data_preprocessing.py
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# 数据目录
def load_data(data_dir):
    dir_list = os.listdir(data_dir)
    dir_list.sort()
    data_df = pd.DataFrame(columns=['path', 'source', 'actor', 'gender', 'emotion'])

    # 读取数据
    count = 0
    for i in dir_list:
        file_list = os.listdir(os.path.join(data_dir, i))
        for f in file_list:
            nm = f.split('.')[0].split('-')
            path = os.path.join(data_dir, i, f)
            src, actor, emotion = int(nm[1]), int(nm[-1]), int(nm[2])
            gender = "female" if int(actor) % 2 == 0 else "male"
            data_df.loc[count] = [path, src, actor, gender, emotion]
            count += 1

    return data_df

def process_labels(data_df):
    label_list = []
    for i in range(len(data_df)):
        emotion_label = {
            2: "_calm", 3: "_happy", 4: "_sad", 5: "_angry", 6: "_fearful"
        }.get(data_df.emotion[i], "_none")
        label_list.append(data_df.gender[i] + emotion_label)
    data_df['label'] = label_list
    return data_df

def extract_features(data_df):
    data = pd.DataFrame(columns=['feature', 'label', 'gender_num'])
    for i in range(len(data_df)):
        X, sample_rate = librosa.load(data_df.path[i], duration=3, sr=22050*2, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        data.loc[i] = [mfccs, data_df.label[i], data_df.gender_num[i]]
    return data

def split_data(data):
    df_features = pd.DataFrame(data['feature'].values.tolist())
    df_features['label'] = data['label']
    df_features['gender_num'] = data['gender_num']
    df_features = df_features.fillna(0)

    X = df_features.drop(['label'], axis=1)
    y = df_features['label']
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test

def encode_labels(y_train, y_test):
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.transform(y_test))
    return y_train, y_test
