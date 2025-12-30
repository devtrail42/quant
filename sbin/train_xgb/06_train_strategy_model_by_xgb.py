import os
import sys
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import math
import numpy as np


def load_from_db(db_path, table_base):
    conn = sqlite3.connect(db_path)

    train_df = pd.read_sql(
        f"SELECT * FROM {table_base}_train",
        conn
    )
    val_df = pd.read_sql(
        f"SELECT * FROM {table_base}_val",
        conn
    )
    test_df = pd.read_sql(
        f"SELECT * FROM {table_base}_test",
        conn
    )

    conn.close()
    return train_df, val_df, test_df

def prepare_xy(df, feat_cols, label_cols, random_state=42):
    # X, Y 중 하나라도 NaN 있으면 제거
    df = (
        df[feat_cols + label_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # 랜덤 셔플
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X = df[feat_cols]
    Y = df[label_cols]

    return X, Y

class Classifier:
    def __init__(self, model_config):
        self.model = XGBClassifier(**model_config)

    def train(self, x_train, y_train, x_val, y_val, num_class=2, average='binary'):
        self.columns = x_train.columns
        # self.model.fit(x_train, y_train, verbose=True,
        #                eval_set=[(x_val, y_val)])
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        y_pred = self.model.predict(x_train)
        predictions = [ round(value) for value in y_pred]
        # evaluate predictions
            
        train_confusion = confusion_matrix(y_train, predictions)
        train_accuracy = accuracy_score(y_train, predictions)
        train_precision = precision_score(y_train, predictions, average=average)
        train_recall = recall_score(y_train, predictions, average=average)
        train_f1 = f1_score(y_train, predictions, average=average)
        #train_auc = roc_auc_score(y_train, predictions, multi_class='ovo', average=average)
        train_auc = train_accuracy
        
        print("오차 행렬")
        print(train_confusion)
        if average == 'binary':
            print(f"정확도: {train_accuracy:.4f}, 정밀도: {train_precision:.4f}, 재현율: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        else:
            print("정확도: ", train_accuracy, ", 정밀도: ", train_precision, "재현율: ", train_recall, "F1: ", train_f1, "AUC: ", train_auc)
        # for i, y_ in enumerate(y_train):
        #     print(y_, y_pred[i])
        
        print ("#############################################################################")
        # make predictions for test data
        y_pred = self.model.predict(x_val)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        test_confusion = confusion_matrix(y_val, predictions)
        test_accuracy = accuracy_score(y_val, predictions)
        test_precision = precision_score(y_val, predictions, average=average)
        test_recall = recall_score(y_val, predictions, average=average)
        test_f1 = f1_score(y_val, predictions, average=average)
        #test_auc = roc_auc_score(y_val, predictions, multi_class='ovo', average=average)
        test_auc = test_accuracy
        
        print("오차 행렬")
        print(test_confusion)
        if average == 'binary':
            print(f"정확도: {test_accuracy:.4f}, 정밀도: {test_precision:.4f}, 재현율: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        else:
            print("정확도: ", test_accuracy, ", 정밀도: ", test_precision, "재현율: ", test_recall, "F1: ", test_f1, "AUC: ", test_auc)

        # for i, y_ in enumerate(y_val):
        #     print(y_, y_pred[i])
        
        feat_score = self.model.feature_importances_
        print(feat_score)
        
        return self.model
    
    def print_importance(self):
        feature_importance = self.model.feature_importances_
        print("feature count : ", len(feature_importance))
        plot_importance(model)
        dic = Counter()
        for col, score in zip(self.columns, feature_importance):
            dic[col] = score
        for col, score in dic.most_common():
            print(col, score)

parser = argparse.ArgumentParser(description='06_train_strategy_model_by_xgb')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/cyb/project/2025/quant")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute60")
parser.add_argument('--target_strategy_feature', type=str, default="low_bb_du")
parser.add_argument('--model_output_dir', type=str, default="var/xgb_model")

args = parser.parse_args()


if __name__ == "__main__":
    
    xgb_dir = os.path.join(args.root_dir, "var/xgb_data")
    table_base = f"xgb_{args.market}_{args.interval}_{args.target_strategy_feature}"
    xgb_db_path = os.path.join(xgb_dir, f"{table_base}.db")
    train_df, val_df, test_df = load_from_db(xgb_db_path, table_base)

    FEATURE_COLS = [f"f{i}" for i in range(1, 23)]
    LABEL_COLS = ["label1", "label2", "label3", "label4"]

    x_train, ys_train = prepare_xy(train_df, FEATURE_COLS, LABEL_COLS)
    x_val, ys_val = prepare_xy(val_df, FEATURE_COLS, LABEL_COLS)
    x_test, ys_test = prepare_xy(test_df, FEATURE_COLS, LABEL_COLS)

    for label_col in LABEL_COLS:
        y_train = ys_train[label_col]
        y_val = ys_val[label_col]
        y_test = ys_test[label_col]

        print(y_train.shape, x_train.shape, y_val.shape, x_val.shape)

        model_config = None

        num_class = 2
        xgb_train_config_path = os.path.join(args.root_dir, 'sbin/train_xgb/binary_classifier_config.json')
        with open(xgb_train_config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            # model_config["eval_metric"] = 'error'
            print('binary config loaded')
        
        # 'smote', 'scale_weight', 'baseline
        strategy = 'scale_weight'
        if 'strategy' in model_config:
            strategy = model_config['strategy']
        del model_config['strategy']

        if 'scale_weight' in strategy:
            max_cnt = 0
            min_cnt = 999999999
            for i in range(num_class):
                i_cnt = len(y_train.loc[y_train == i])
                max_cnt = max(max_cnt, i_cnt)
                min_cnt = min(min_cnt, i_cnt)
            model_config['scale_pos_weight'] = max_cnt/min_cnt
            print("scale_pos_weight : ", model_config['scale_pos_weight'])
        elif 'smote' in strategy:
            smote = SMOTE()
            x_train, y_train = smote.fit_resample(x_train, y_train)
            print('after resample : ', len(x_train), len(y_train))

        config_string = json.dumps(model_config)
        classifier = Classifier(model_config)
        average = 'binary'
        model = classifier.train(x_train, y_train, x_val, y_val, num_class=num_class, average=average)
        classifier.print_importance()
        print(config_string)
        model_out_dir = os.path.join(args.root_dir, args.model_output_dir)
        os.makedirs(model_out_dir, exist_ok=True)
        
        model_output_path = os.path.join(model_out_dir,  table_base + '_' + label_col + '.classifier')
        model.save_model(model_output_path)

