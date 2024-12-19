# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from data_preprocessing import load_data, process_labels, extract_features, split_data, encode_labels
from model import build_model

# 配置
data_dir = './Audio_Speech_Actors_01-24'

# 数据处理
data_df = load_data(data_dir)
data_df = process_labels(data_df)
data_df['gender_num'] = data_df['gender'].map({'male': 0, 'female': 1})

data = extract_features(data_df)
X_train, X_test, y_train, y_test = split_data(data)
y_train, y_test = encode_labels(y_train, y_test)

# 构建模型
model = build_model((X_train.shape[1], 1))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 回调函数
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
mcp_save = ModelCheckpoint('./logs/Data_with_gender_prompt.keras', save_best_only=True, monitor='val_loss', mode='min')
csv_logger = CSVLogger('./logs/training_log_with_gender_f1.csv', append=True)

# 扩展性别特征
prompt_train = X_train[['gender_num']].values
prompt_test = X_test[['gender_num']].values
prompt_train_expanded = np.repeat(prompt_train[:, np.newaxis, :], X_train.shape[1], axis=1)
prompt_test_expanded = np.repeat(prompt_test[:, np.newaxis, :], X_test.shape[1], axis=1)

# 训练模型
cnnhistory = model.fit([X_train, prompt_train_expanded], y_train,
                       batch_size=16, epochs=100,
                       validation_data=([X_test, prompt_test_expanded], y_test),
                       callbacks=[mcp_save, lr_reduce, csv_logger])

# 绘制损失曲线
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('Model Loss with Gender Prompt Learning')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 评估模型
loaded_model = tf.keras.models.load_model('./logs/Data_with_gender_prompt.keras')
score = loaded_model.evaluate([X_test, prompt_test_expanded], y_test, verbose=0)
print(f"Accuracy: {score[1]*100:.2f}%")

# 混淆矩阵
preds = loaded_model.predict([X_test, prompt_test_expanded])
preds_labels = preds.argmax(axis=1)
y_true_labels = y_test.argmax(axis=1)
c_matrix = confusion_matrix(y_true_labels, preds_labels)
print("Confusion Matrix:\n", c_matrix)

# 可视化混淆矩阵
class_names = ['Class 1', 'Class 2']  # 根据实际分类调整
df_cm = pd.DataFrame(c_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
