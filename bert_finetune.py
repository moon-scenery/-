# bert_finetune.py
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

# ============================== 配置参数 ==============================
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
    EPOCHS = 3
    LR = 2e-5
    N_FOLDS = 5
    USE_GPU = True
    THRESHOLD = 0.65             # 正面概率大于此值才判为正面（可调）

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
        df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
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
        y = self.df['label']
        if y.isnull().any():
            y = y.dropna()
            X = X.loc[y.index]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        return X_train, X_test, y_train, y_test

# ============================== BERT微调模型（固定种子 + 阈值判定 + 资源监控）==============================
class BertFinetuneModel:
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed
        self.set_seed(seed)
        self.device = self.get_device()
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
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

    def create_dataset(self, texts, labels):
        class FineTuneDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                text = str(self.texts.iloc[idx])
                encoding = self.tokenizer(
                    text, max_length=self.max_len, padding='max_length',
                    truncation=True, return_tensors='pt',
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
                }
        return FineTuneDataset(texts, labels, self.tokenizer, self.config.MAX_LEN)

    def train_cv(self, X_train, y_train):
        print("\n" + "="*50)
        print(f"[Seed {self.seed}] 开始 {self.config.N_FOLDS} 折交叉验证 BERT微调（10:1 类别权重）")
        print("="*50)

        self.start_monitor()

        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.seed)

        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [],
                        'f1': [], 'negative_recall': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n--- Fold {fold+1}/{self.config.N_FOLDS} ---")

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            train_dataset = self.create_dataset(X_fold_train, y_fold_train)
            val_dataset = self.create_dataset(X_fold_val, y_fold_val)

            model = BertForSequenceClassification.from_pretrained(
                self.config.BERT_MODEL_NAME, num_labels=2
            ).to(self.device)

            # 负类权重提高到 10:1
            class_weights = torch.tensor([10.0, 1.0]).to(self.device)

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=self.config.EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                logging_dir='./logs',
                logging_steps=100,
                save_strategy='no',
                report_to='none'
            )

            trainer = WeightedTrainer(
                model=model, args=training_args, train_dataset=train_dataset,
            )
            trainer.train()

            # 验证集评估（使用阈值判定）
            predictions = trainer.predict(val_dataset)
            logits = predictions.predictions
            probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # 正面概率
            y_pred = (probs >= self.config.THRESHOLD).astype(int)
            y_true = y_fold_val.values

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, pos_label=1)
            rec = recall_score(y_true, y_pred, pos_label=1)
            f1_val = f1_score(y_true, y_pred, pos_label=1)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1_val)
            fold_metrics['negative_recall'].append(neg_rec)

            print(f"Fold {fold+1} - 准确率: {acc:.4f}, F1: {f1_val:.4f}, 负面召回率: {neg_rec:.4f} (阈值={self.config.THRESHOLD})")

        # 计算平均值和标准差
        self.cv_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'negative_recall']:
            values = fold_metrics[metric]
            self.cv_results[metric + '_mean'] = np.mean(values)
            self.cv_results[metric + '_std'] = np.std(values)

        print("\n" + "="*50)
        print(f"[Seed {self.seed}] BERT微调 {self.config.N_FOLDS} 折交叉验证结果")
        print("="*50)
        print(f"准确率:     {self.cv_results['accuracy_mean']:.4f} (±{self.cv_results['accuracy_std']:.4f})")
        print(f"精确率:     {self.cv_results['precision_mean']:.4f} (±{self.cv_results['precision_std']:.4f})")
        print(f"召回率:     {self.cv_results['recall_mean']:.4f} (±{self.cv_results['recall_std']:.4f})")
        print(f"F1分数:     {self.cv_results['f1_mean']:.4f} (±{self.cv_results['f1_std']:.4f})")
        print(f"负面召回率: {self.cv_results['negative_recall_mean']:.4f} (±{self.cv_results['negative_recall_std']:.4f})")

        self.end_monitor()
        return self.cv_results

    def train_final_and_test(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*50)
        print(f"[Seed {self.seed}] 在全部训练集上训练最终模型...")
        print("="*50)

        self.start_monitor()

        train_dataset = self.create_dataset(X_train, y_train)

        model = BertForSequenceClassification.from_pretrained(
            self.config.BERT_MODEL_NAME, num_labels=2
        ).to(self.device)

        # 负类权重 10:1
        class_weights = torch.tensor([8.0, 1.0]).to(self.device)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config.EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            logging_dir='./logs',
            logging_steps=100,
            save_strategy='no',
            report_to='none'
        )

        trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()

        log_history = trainer.state.log_history
        steps, losses = [], []
        for log in log_history:
            if 'loss' in log and 'step' in log:
                steps.append(log['step'])
                losses.append(log['loss'])
        
        if len(steps) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(steps, losses, marker='o', markersize=2, linewidth=1.5)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title(f'Training Loss (Seed {self.seed})')
            plt.grid(True, alpha=0.3)
            plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("训练 loss 图已保存至 training_loss.png")
        else:
            print("警告：未获取到训练 loss 记录，无法绘图。")


        
        test_dataset = self.create_dataset(X_test, y_test)
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # 正面概率
        y_true = y_test.values

        # 使用阈值分类
        y_pred = (probs >= self.config.THRESHOLD).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1)
        rec = recall_score(y_true, y_pred, pos_label=1)
        f1_val = f1_score(y_true, y_pred, pos_label=1)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        neg_rec = tn / (tn + fp)

        print(f"\n测试集最终结果 (阈值={self.config.THRESHOLD}):")
        print(f"准确率: {acc:.4f}, F1: {f1_val:.4f}, 负面召回率: {neg_rec:.4f}")

        self.results = {
            'model_type': 'BERT微调',
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1_val,
            'negative_recall': neg_rec, 'confusion_matrix': cm
        }

        self.end_monitor()
        return self.results

# ============================== 主流程（固定种子 456，阈值可调）==============================
if __name__ == "__main__":
    print("="*50)
    print("电商评论情感分析系统 - BERT微调 (固定种子 456 + 阈值判定)")
    print("="*50)

    # 固定随机种子
    config.RANDOM_STATE = 456

    processor = DataProcessor(config)
    X_train, X_test, y_train, y_test = processor.prepare_data()

    model = BertFinetuneModel(config, seed=456)

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

    joblib.dump(record, 'bert_finetune_seed456_results.pkl')

    # 打印资源效率比 RE
    acc = test_results['accuracy']
    time_sec = model.train_time
    gpu_mem = model.gpu_memory
    if gpu_mem > 0 and time_sec > 0:
        re_value = acc / (time_sec * gpu_mem)
        print(f"\n资源效率比 (RE): {re_value:.6f}")
    else:
        re_value = None

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
    joblib.dump(final_summary, 'bert_finetune_summary.pkl')

    print("\n固定种子 456 训练完成，结果已保存。")