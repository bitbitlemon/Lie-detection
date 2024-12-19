# utils.py
import numpy as np
from sklearn.metrics import f1_score

# F1分数计算
def compute_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')
