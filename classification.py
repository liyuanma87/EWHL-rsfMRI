import os
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pylab import *
import copy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_input_file(filename, legacy_path=None):
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(os.getcwd(), filename),
    ]
    if legacy_path:
        candidates.append(legacy_path)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"找不到输入文件: {filename}\n"
        + "已检查以下位置:\n  - "
        + "\n  - ".join(candidates)
        + "\n请将该文件放到本 Python 脚本同一文件夹，或修改代码中的路径。"
    )

# 1. 读取Excel文件，提取特征（第3列到第145列，共143个特征）
excel_path = _find_input_file(
    '884_subjects_143features.xlsx',
    legacy_path='/Users/maliyuan/Desktop/Code/884_subjects_143features.xlsx'
)

df_full = pd.read_excel(excel_path)

feature_matrix = df_full.iloc[:, 2:145].values  # 提取884×143的特征矩阵
# print(f"特征矩阵形状: {feature_matrix.shape}")
# print(f"特征数量: {feature_matrix.shape[1]}")

column_means = np.mean(feature_matrix, axis=0)
# print(column_means)
#
combat_path = _find_input_file(
    'matlab_combat_143.txt',
    legacy_path='/Users/maliyuan/Desktop/Code/matlab_combat_143.txt'
)
txt=np.loadtxt(combat_path)


combat_feature_matrix=np.zeros((884,143))

nonzero_column_means=[]
for i in column_means:
    if i !=0:
        nonzero_column_means.append(i)
# print(len(nonzero_column_means))
for i in range(143):
    for j in range(884):
        combat_feature_matrix[j][i]=txt[j][i]+nonzero_column_means[i]
combat_out = os.path.join(SCRIPT_DIR, 'combat_feature_matrix143')
np.savetxt(combat_out, combat_feature_matrix)
combat_feature_matrix=np.loadtxt(combat_out)
# print(combat_feature_matrix)
dataset = pd.DataFrame(combat_feature_matrix)
metadata_path = _find_input_file(
    '884data.xlsx',
    legacy_path='/Users/maliyuan/Desktop/Code/884data.xlsx'
)
data=pd.read_excel(metadata_path)
Y1= data['AGE']
#print(Y1)
Y=[]
for i in range(884):
    Y.append(Y1[i])
mean_Y=np.mean(Y)
#for column in dataset.columns:
lst1,lst2=[],[]
for i in range(143):
    X=[]
    for j in range(884):
         X.append(dataset[i][j])

    up=0
    down1=0
    for k in range(884):
        up+=X[k]*(Y[k]-mean_Y)
        down1+=(Y[k]**2)

    w=up/(down1-(((np.sum(Y))**2)/884))
    b=0
    for l in range(884):
        b+= (X[l]-w*Y[l])

    lst1.append(w)
    lst2.append(b/884)
# print(lst1)
# print(lst2)
residual_numpy=np.zeros((884,143))
residual=residual_numpy.tolist()
for i in range(143):
    for j in range(884):
        residual[j][i]=dataset[i][j]-lst1[i]*Y1[j]-lst2[i]
#print(np.asarray(residual))
residual_out = os.path.join(SCRIPT_DIR, '884143residual_matrix和年龄回归')
np.savetxt(residual_out, np.asarray(residual))
matrix111=np.loadtxt(residual_out)


class_feature_matrix=np.zeros((884,143))
ar=np.zeros(884)
import random
number=list(range(884))
random.shuffle(number)
for i in range(884):
    class_feature_matrix[i]=matrix111[number[i]]
    ar[i]=data['DXGROUP'][number[i]]
    if ar[i]==2:
        ar[i]=0
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold

# ======================== 20次重复10折交叉验证 ========================
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score as sklearn_f1_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

N_REPEATS = 20
N_SPLITS = 10

data, target = class_feature_matrix[:], ar[:]
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = round(data[i][j], 2)
data = np.asarray(data)
target = np.asarray(target)


def evaluate_model(model_name, model_builder):
    
    repeat_accuracy = []
    repeat_specificity = []
    repeat_sensitivity = []
    repeat_f1 = []
    repeat_auc = []
    repeat_train_accuracy = []
    feature_importance = np.zeros(data.shape[1])

    for repeat in range(N_REPEATS):
        
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)

        fold_accuracy = []
        fold_specificity = []
        fold_sensitivity = []
        fold_f1 = []
        fold_auc = []
        fold_train_accuracy = []

        for train_index, test_index in kf.split(data):
            model = model_builder(repeat)
            model.fit(data[train_index], target[train_index])

            y_pred = model.predict(data[test_index])
            y_true = target[test_index]

            fold_accuracy.append(model.score(data[test_index], y_true))
            fold_train_accuracy.append(model.score(data[train_index], target[train_index]))

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fold_specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            fold_sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fold_f1.append(sklearn_f1_score(y_true, y_pred, zero_division=0))

        
            target_exchange = []
            predict_exchange = []
            for item in target[test_index]:
                if item == 2:
                    item = 0
                    target_exchange.append(item)
                else:
                    target_exchange.append(item)
            for item in model.predict_proba(list(data[test_index])):
                predict_exchange.append(item[1])
            y_label = np.array(target_exchange)
            y_pred_auc = np.array(predict_exchange)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fpr[0], tpr[0], _ = roc_curve(y_label, y_pred_auc)
            roc_auc[0] = auc(fpr[0], tpr[0])
            fold_auc.append(roc_auc[0])

            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_

        # 先求每一次10折交叉验证的平均值
        repeat_accuracy.append(np.mean(fold_accuracy))
        repeat_specificity.append(np.mean(fold_specificity))
        repeat_sensitivity.append(np.mean(fold_sensitivity))
        repeat_f1.append(np.mean(fold_f1))
        repeat_auc.append(np.mean(fold_auc))
        repeat_train_accuracy.append(np.mean(fold_train_accuracy))

    def report(metric_name, values):
        values = np.asarray(values)
        print(f'{metric_name}平均值：{np.mean(values):.10f}')
        print(f'{metric_name}标准差：{np.std(values, ddof=1):.10f}')

    print('\n' + '=' * 24 + f' {model_name} ' + '=' * 24)
    report('测试集准确率', repeat_accuracy)
    report('specificity', repeat_specificity)
    report('sensitivity', repeat_sensitivity)
    report('F1-score', repeat_f1)
    report('AUC', repeat_auc)
    report('训练集准确率', repeat_train_accuracy)

    # 20×10=200个模型的平均特征重要性
    mean_feature_importance = feature_importance / (N_REPEATS * N_SPLITS)
    print('平均特征重要性：')
    print(mean_feature_importance.tolist())

    return {
        'accuracy': repeat_accuracy,
        'specificity': repeat_specificity,
        'sensitivity': repeat_sensitivity,
        'f1': repeat_f1,
        'auc': repeat_auc,
        'train_accuracy': repeat_train_accuracy,
        'feature_importance': mean_feature_importance
    }


rf_results = evaluate_model(
    'Random Forest',
    lambda repeat: RandomForestClassifier(
        max_depth=5,
        n_estimators=100,
        random_state=40 + repeat,
        criterion='entropy'
    )
)

gbdt_results = evaluate_model(
    'GBDT',
    lambda repeat: GradientBoostingClassifier(
        max_depth=2,
        n_estimators=100,
        learning_rate=0.06,
        random_state=42 + repeat
    )
)

xgb_results = evaluate_model(
    'XGBoost',
    lambda repeat: xgb.XGBClassifier(
        max_depth=2,
        n_estimators=110,
        learning_rate=0.0844,
        random_state=42 + repeat,
        eval_metric='logloss'
    )
)
