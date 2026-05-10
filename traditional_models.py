# traditional_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import jieba
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

# ============================== 配置参数 ==============================
class Config:
    DATA_PATH = 'jdc_drop_duplicatesdb.xlsx'
    STOPWORDS_PATH = 'hit_stopwords.txt'
    THRESHOLD = 0.45   # 比0.5低，可调（0.4~0.45之间）
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
    TFIDF_MAX_FEATURES = 5000
    N_FOLDS = 5
    MODELS = ['LR', 'SVM', 'RF']

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
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, 
            stratify=y
        )
        return X_train, X_test, y_train, y_test

# ============================== 传统模型（5 折 CV + SMOTE + 时间记录）==============================
class TraditionalModel:
    def __init__(self, model_type, config):
        self.model_type = model_type
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES)
        self.model = None
        self.cv_results = {}
        self.train_time = 0.0
        self.results = {}
    
    def train_cv(self, X_train, y_train):
        """5 折交叉验证，每折内使用 SMOTE 平衡训练数据"""
        print(f"\n{'='*50}")
        print(f"[{self.model_type}] 开始 {self.config.N_FOLDS} 折交叉验证 (SMOTE)")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # TF-IDF 向量化整个训练集（在 CV 外做一次，避免重复 fit）
        X_vec = self.vectorizer.fit_transform(X_train)
        y = y_train
        
        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [],
                        'f1': [], 'negative_recall': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_vec, y)):
            print(f"\n--- Fold {fold+1}/{self.config.N_FOLDS} ---")
            X_tr_fold = X_vec[train_idx]
            y_tr_fold = y.iloc[train_idx]
            X_val_fold = X_vec[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # SMOTE 过采样
            smote = SMOTE(random_state=self.config.RANDOM_STATE)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_fold, y_tr_fold)
            
            # 根据模型类型创建分类器
            if self.model_type == 'LR':
                model = LogisticRegression(max_iter=1000, solver='saga', penalty='l2')
            elif self.model_type == 'SVM':
                model = SVC(kernel='linear', probability=True)
            elif self.model_type == 'RF':
                model = RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
            
            model.fit(X_tr_res, y_tr_res)
            y_pred = model.predict(X_val_fold)
            
            # 评估
            y_true_num = y_val_fold.map({'negative': 0, 'positive': 1})
            y_pred_num = np.array([0 if p == 'negative' else 1 for p in y_pred])
            
            acc = accuracy_score(y_true_num, y_pred_num)
            prec = precision_score(y_true_num, y_pred_num, pos_label=1)
            rec = recall_score(y_true_num, y_pred_num, pos_label=1)
            f1 = f1_score(y_true_num, y_pred_num, pos_label=1)
            cm = confusion_matrix(y_true_num, y_pred_num)
            tn, fp, fn, tp = cm.ravel()
            neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1)
            fold_metrics['negative_recall'].append(neg_rec)
            
            print(f"Fold {fold+1} - 准确率: {acc:.4f}, F1: {f1:.4f}, 负面召回率: {neg_rec:.4f}")
        
        # 计算 CV 均值和标准差
        self.cv_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'negative_recall']:
            values = fold_metrics[metric]
            self.cv_results[metric + '_mean'] = np.mean(values)
            self.cv_results[metric + '_std'] = np.std(values)
        
        cv_time = time.time() - start_time
        # 注意：这里的时间仅为 CV 训练时间，不包含之后的最终训练
        self.cv_train_time = cv_time
        
        print("\n" + "="*50)
        print(f"[{self.model_type}] {self.config.N_FOLDS} 折交叉验证结果")
        print("="*50)
        print(f"准确率:     {self.cv_results['accuracy_mean']:.4f} (±{self.cv_results['accuracy_std']:.4f})")
        print(f"精确率:     {self.cv_results['precision_mean']:.4f} (±{self.cv_results['precision_std']:.4f})")
        print(f"召回率:     {self.cv_results['recall_mean']:.4f} (±{self.cv_results['recall_std']:.4f})")
        print(f"F1分数:     {self.cv_results['f1_mean']:.4f} (±{self.cv_results['f1_std']:.4f})")
        print(f"负面召回率: {self.cv_results['negative_recall_mean']:.4f} (±{self.cv_results['negative_recall_std']:.4f})")
        
        return self.cv_results
    
    def train_final_and_test(self, X_train, y_train, X_test, y_test):
        """在全部训练集上训练最终模型，并在测试集上评估（包含计时）"""
        print(f"\n{'='*50}")
        print(f"[{self.model_type}] 在全部训练集上训练最终模型...")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 重新向量化（使用新的训练集，但为了保持一致，可用之前 fit 的 vectorizer）
        # 注意：CV 中已经 fit 过 vectorizer，这里可以重用 transform
        X_train_vec = self.vectorizer.transform(X_train)   # 已经 fit 过
        X_test_vec = self.vectorizer.transform(X_test)
        
        # SMOTE 平衡全部训练集
        smote = SMOTE(random_state=self.config.RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_train_vec, y_train)
        
        # 训练最终模型
        if self.model_type == 'LR':
            self.final_model = LogisticRegression(max_iter=1000, solver='saga', penalty='l2')
        elif self.model_type == 'SVM':
            self.final_model = SVC(kernel='linear', probability=True)
        elif self.model_type == 'RF':
            self.final_model = RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
        
        self.final_model.fit(X_tr_res, y_tr_res)
        
        #y_pred = self.final_model.predict(X_test_vec)
        if hasattr(self.final_model, "predict_proba"):
            proba = self.final_model.predict_proba(X_test_vec)[:, 1]   # 正面概率
        else:
            decision = self.final_model.decision_function(X_test_vec)
            proba = 1 / (1 + np.exp(-decision))   # sigmoid
        y_pred = np.where(proba >= self.config.THRESHOLD, 'positive', 'negative')

        
        
        y_true_num = y_test.map({'negative': 0, 'positive': 1})
        y_pred_num = np.array([0 if p == 'negative' else 1 for p in y_pred])
        
        acc = accuracy_score(y_true_num, y_pred_num)
        prec = precision_score(y_true_num, y_pred_num, pos_label=1)
        rec = recall_score(y_true_num, y_pred_num, pos_label=1)
        f1 = f1_score(y_true_num, y_pred_num, pos_label=1)
        cm = confusion_matrix(y_true_num, y_pred_num)
        tn, fp, fn, tp = cm.ravel()
        neg_rec = tn / (tn + fp)
        
        self.train_time = time.time() - start_time
        
        print(f"\n测试集最终结果:")
        print(f"准确率: {acc:.4f}, F1: {f1:.4f}, 负面召回率: {neg_rec:.4f}")
        print(f"训练耗时（最终模型）: {self.train_time:.2f} 秒")
        
        self.results = {
            'model_type': self.model_type,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'negative_recall': neg_rec,
            'confusion_matrix': cm,
            'train_time_sec': self.train_time,
            'RE': 'N/A'      # 传统模型不使用 GPU
        }
        return self.results

# ============================== 主流程 ==============================
if __name__ == "__main__":
    print("="*50)
    print("电商评论情感分析系统 - 传统模型 (5折CV + SMOTE + 资源监控)")
    print("="*50)
    
    processor = DataProcessor(config)
    X_train, X_test, y_train, y_test = processor.prepare_data()
    
    all_summaries = {}
    
    for model_type in config.MODELS:
        print(f"\n{'#'*50}")
        print(f"### 训练 {model_type} 模型")
        print(f"{'#'*50}")
        
        model = TraditionalModel(model_type, config)
        
        # 5 折交叉验证
        cv_results = model.train_cv(X_train, y_train)
        
        # 最终测试集评估
        test_results = model.train_final_and_test(X_train, y_train, X_test, y_test)
        
        # 保存每个模型的详细结果
        model_record = {
            'model_type': model_type,
            'cv_results': cv_results,
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results['f1'],
            'test_negative_recall': test_results['negative_recall'],
            'train_time_sec': test_results['train_time_sec'],
            'gpu_memory_MB': None,
            'RE': 'N/A',
            'confusion_matrix': test_results['confusion_matrix']
        }
        joblib.dump(model_record, f'{model_type}_seed456_results.pkl')
        
        # 生成符合汇总报告格式的 summary
        summary = {
            'model_type': model_type,
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],   # ← 新增
            'recall': test_results['recall'],         # ← 新增
            'f1': test_results['f1'],
            'negative_recall': test_results['negative_recall'],
            'train_time_sec': test_results['train_time_sec'],
            'gpu_memory_MB': None,
            'RE': 'N/A',
            'confusion_matrix': test_results['confusion_matrix']
        }
        all_summaries[model_type] = summary
    
    # 保存汇总文件（供 summary_report.py 使用）
    joblib.dump(all_summaries, 'traditional_models_summary.pkl')
    
    print("\n所有传统模型训练完成，结果已保存至 traditional_models_summary.pkl")