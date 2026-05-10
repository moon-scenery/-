# bert_lr_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import jieba
import warnings
import random
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

class Config:
    DATA_PATH = 'jdc_drop_duplicatesdb.xlsx'
    STOPWORDS_PATH = 'hit_stopwords.txt'

    CUSTOM_STOPWORDS = [
        '买了', '已经', '感觉', '一个', '这个', '那个',
        '比较', '算是', '可能', '好像', '似乎', '大概',
        '今天', '昨天', '刚刚', '现在', '之前', '以后',
        '收到', '到了', '到货', '发货', '下单',
        '淘宝', '京东', '拼多多', '天猫',
        '包邮', '优惠', '活动',
        '啊', '呀', '呢', '吧', '啦', '哦', '嗯',
        '哈', '哟', '唉', '呃', '嘛',
        '看看', '问问', '想要',
        '退货', '换货', '咨询', '收藏', '点赞', '关注'
    ]

    TEST_SIZE = 0.2
    RANDOM_STATE = 456          # 固定种子
    BERT_MODEL_NAME = './bert-base-chinese-local/tiansz/bert-base-chinese'
    MAX_LEN = 128
    BATCH_SIZE = 16
    N_FOLDS = 5
    USE_GPU = True
    THRESHOLD = 0.55            # 正面概率大于此值才判为正面

config = Config()

# ============================== 数据预处理 ==============================
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.df = self.load_data()
        self.stopwords = self.load_stopwords()

    def load_data(self):
        print("\n[1/4] 正在加载数据...")
        df = pd.read_excel(self.config.DATA_PATH)
        print(f"原始数据量: {len(df)}条")
        df['sentiment'] = df['评分（总分5分）(score)'].apply(
            lambda x: 'positive' if x >= 4 else 'negative'
        )
        return df

    def load_stopwords(self):
        print("\n[2/4] 正在加载停用词表...")
        with open(self.config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
        stopwords.update(self.config.CUSTOM_STOPWORDS)
        print(f"停用词总量: {len(stopwords)}")
        return stopwords

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s]', '', text)
        words = jieba.lcut(text)
        words = [word for word in words if word not in self.stopwords and len(word) > 1]
        return ' '.join(words)

    def prepare_data(self):
        print("\n[3/4] 正在预处理文本数据...")
        tqdm.pandas(desc="文本预处理进度")
        self.df['cleaned_content'] = self.df['评价内容(content)'].progress_apply(self.preprocess_text)

        print("\n[4/4] 正在划分数据集...")
        X = self.df['cleaned_content']
        y = self.df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

# ============================== BERT+LR 模型（固定种子 + 资源监控 + 阈值判定）==============================
class BertLRModel:
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.set_seed(seed)
        self.device = self.get_device()
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        self.bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(self.device)
        self.cv_results = {}
        self.train_time = 0.0
        self.gpu_memory = 0.0

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_device(self):
        if config.USE_GPU and torch.cuda.is_available():
            print("使用GPU加速")
            return torch.device('cuda')
        print("使用CPU")
        return torch.device('cpu')

    def start_monitor(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()

    def end_monitor(self):
        self.train_time = time.time() - self.start_time
        if self.device.type == 'cuda':
            self.gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            self.gpu_memory = 0.0

    def create_bert_dataset(self, texts):
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_len = max_len
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                text = str(self.texts.iloc[idx])
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
        return TextDataset(texts, self.tokenizer, self.config.MAX_LEN)

    def extract_bert_features(self, texts):
        dataset = self.create_bert_dataset(texts)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        self.bert_model.eval()
        features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取BERT特征"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.bert_model(input_ids, attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                selected_layers = [hidden_states[i] for i in range(-4, 0)]
                pooled_layers = torch.mean(torch.stack(selected_layers), dim=0)
                cls_embedding = pooled_layers[:, 0, :].cpu().numpy()
                features.extend(cls_embedding)
        return np.array(features)

    def train_cv(self, X_train, y_train):
        print("\n" + "="*50)
        print(f"[Seed {self.seed}] 开始 {self.config.N_FOLDS} 折交叉验证 BERT+LR (阈值={self.config.THRESHOLD})")
        print("="*50)

        self.start_monitor()

        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.seed)

        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [],
                        'f1': [], 'negative_recall': [], 'cm_list': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n--- Fold {fold+1}/{self.config.N_FOLDS} ---")

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # 提取 BERT 特征
            print("提取训练集特征...")
            X_tr_bert = self.extract_bert_features(X_fold_train)
            print("提取验证集特征...")
            X_val_bert = self.extract_bert_features(X_fold_val)

            # SMOTE 平衡训练集
            smote = SMOTE(random_state=self.seed)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_bert, y_fold_train)

            # 训练 LR（无类别权重）
            lr = LogisticRegression(max_iter=1000, C=0.1, solver='saga', penalty='l2')
            lr.fit(X_tr_res, y_tr_res)

            # 预测概率并应用阈值
            if hasattr(lr, "predict_proba"):
                proba = lr.predict_proba(X_val_bert)[:, 1]   # 正面概率
            else:
                # 备用：使用 decision_function + sigmoid
                decision = lr.decision_function(X_val_bert)
                proba = 1 / (1 + np.exp(-decision))
            y_pred = np.where(proba >= self.config.THRESHOLD, 'positive', 'negative')

            y_true_num = y_fold_val.map({'negative': 0, 'positive': 1})
            y_pred_num = np.array([0 if p == 'negative' else 1 for p in y_pred])

            acc = accuracy_score(y_true_num, y_pred_num)
            prec = precision_score(y_true_num, y_pred_num, pos_label=1)
            rec = recall_score(y_true_num, y_pred_num, pos_label=1)
            f1_val = f1_score(y_true_num, y_pred_num, pos_label=1)
            cm = confusion_matrix(y_true_num, y_pred_num)
            tn, fp, fn, tp = cm.ravel()
            neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1_val)
            fold_metrics['negative_recall'].append(neg_rec)
            fold_metrics['cm_list'].append(cm)

            print(f"Fold {fold+1} - 准确率: {acc:.4f}, F1: {f1_val:.4f}, 负面召回率: {neg_rec:.4f}")

        # 计算平均值和标准差
        self.cv_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'negative_recall']:
            values = fold_metrics[metric]
            self.cv_results[metric + '_mean'] = np.mean(values)
            self.cv_results[metric + '_std'] = np.std(values)

        print("\n" + "="*50)
        print(f"[Seed {self.seed}] BERT+LR {self.config.N_FOLDS} 折交叉验证结果")
        print("="*50)
        print(f"准确率:     {self.cv_results['accuracy_mean']:.4f} (±{self.cv_results['accuracy_std']:.4f})")
        print(f"精确率:     {self.cv_results['precision_mean']:.4f} (±{self.cv_results['precision_std']:.4f})")
        print(f"召回率:     {self.cv_results['recall_mean']:.4f} (±{self.cv_results['recall_std']:.4f})")
        print(f"F1分数:     {self.cv_results['f1_mean']:.4f} (±{self.cv_results['f1_std']:.4f})")
        print(f"负面召回率: {self.cv_results['negative_recall_mean']:.4f} (±{self.cv_results['negative_recall_std']:.4f})")

        self.end_monitor()
        return self.cv_results

    def train_final_and_test(self, X_train, y_train, X_test, y_test):
        """在全部训练集上训练最终模型，并在测试集上评估"""
        print("\n" + "="*50)
        print(f"[Seed {self.seed}] 在全部训练集上训练最终模型...")
        print("="*50)

        self.start_monitor()

        X_train_bert = self.extract_bert_features(X_train)
        smote = SMOTE(random_state=self.seed)
        X_tr_res, y_tr_res = smote.fit_resample(X_train_bert, y_train)

        self.final_model = LogisticRegression(max_iter=1000, C=0.1, solver='saga', penalty='l2')
        self.final_model.fit(X_tr_res, y_tr_res)

        X_test_bert = self.extract_bert_features(X_test)
        proba = self.final_model.predict_proba(X_test_bert)[:, 1]
        y_pred = np.where(proba >= self.config.THRESHOLD, 'positive', 'negative')

        y_true_num = y_test.map({'negative': 0, 'positive': 1})
        y_pred_num = np.array([0 if p == 'negative' else 1 for p in y_pred])

        acc = accuracy_score(y_true_num, y_pred_num)
        prec = precision_score(y_true_num, y_pred_num, pos_label=1)
        rec = recall_score(y_true_num, y_pred_num, pos_label=1)
        f1_val = f1_score(y_true_num, y_pred_num, pos_label=1)
        cm = confusion_matrix(y_true_num, y_pred_num)
        tn, fp, fn, tp = cm.ravel()
        neg_rec = tn / (tn + fp)

        print(f"\n测试集最终结果 (阈值={self.config.THRESHOLD}):")
        print(f"准确率: {acc:.4f}, F1: {f1_val:.4f}, 负面召回率: {neg_rec:.4f}")

        self.end_monitor()

        self.results = {
            'model_type': 'BERT+LR',
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1_val,
            'negative_recall': neg_rec, 'confusion_matrix': cm
        }
        return self.results

# ============================== 主流程 ==============================
if __name__ == "__main__":
    print("="*50)
    print("电商评论情感分析系统 - BERT+LR (固定种子 456 + 阈值判定 + 资源监控)")
    print("="*50)

    processor = DataProcessor(config)
    X_train, X_test, y_train, y_test = processor.prepare_data()

    model = BertLRModel(config, seed=456)

    # 5 折交叉验证
    cv_results = model.train_cv(X_train, y_train)

    # 最终测试集评估
    test_results = model.train_final_and_test(X_train, y_train, X_test, y_test)

    # 保存结果
    record = {
        'seed': 456,
        'cv_results': cv_results,
        'test_accuracy': test_results['accuracy'],
        'test_f1': test_results['f1'],
        'test_negative_recall': test_results['negative_recall'],
        'train_time_sec': model.train_time,
        'gpu_memory_MB': model.gpu_memory,
        'confusion_matrix': test_results['confusion_matrix']
    }
    joblib.dump(record, 'bert_lr_seed456_results.pkl')

    # 计算资源效率比 RE
    acc = test_results['accuracy']
    time_sec = model.train_time
    gpu_mem = model.gpu_memory
    if gpu_mem > 0 and time_sec > 0:
        re_value = acc / (time_sec * gpu_mem)
        print(f"\n资源效率比 (RE): {re_value:.6f}")
    else:
        re_value = None
        print("\nGPU显存或训练时间记录异常，RE不可用。")

    final_summary = {
        'model_type': 'BERT微调',    # 或 'BERT+LR'
        'accuracy': acc,
        'precision': test_results['precision'],   # ← 加上
        'recall': test_results['recall'],         # ← 加上
        'f1': test_results['f1'],
        'negative_recall': test_results['negative_recall'],
        'train_time_sec': time_sec,
        'gpu_memory_MB': gpu_mem,
        'RE': re_value,
        'confusion_matrix': test_results['confusion_matrix']}
    joblib.dump(final_summary, 'bert_lr_summary.pkl')

    print("\nBERT+LR 训练完成，结果已保存。")