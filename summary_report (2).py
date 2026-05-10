# summary_report.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    results = {}

    # 传统模型（LR, SVM, RF）
    # 优先使用新版 traditional_models_summary.pkl
    if os.path.exists('traditional_models_summary.pkl'):
        trad_results = joblib.load('traditional_models_summary.pkl')
        for model, res in trad_results.items():
            results[model] = res
            results[model].setdefault('train_time_sec', None)
            results[model].setdefault('gpu_memory_MB', None)
            results[model].setdefault('RE', None)
    elif os.path.exists('traditional_models_results.pkl'):
        # 旧版回退
        trad_results = joblib.load('traditional_models_results.pkl')
        for model, res in trad_results.items():
            results[model] = res
            results[model].setdefault('train_time_sec', None)
            results[model].setdefault('gpu_memory_MB', None)
            results[model].setdefault('RE', None)

    # BERT+LR（新版 summary）
    if os.path.exists('bert_lr_summary.pkl'):
        bert_lr = joblib.load('bert_lr_summary.pkl')
        results['BERT+LR'] = bert_lr
        # 确保必要字段存在
        results['BERT+LR'].setdefault('train_time_sec', None)
        results['BERT+LR'].setdefault('gpu_memory_MB', None)
        results['BERT+LR'].setdefault('RE', None)

    # BERT微调（新版 summary）
    if os.path.exists('bert_finetune_summary.pkl'):
        bert_ft = joblib.load('bert_finetune_summary.pkl')
        results['BERT微调'] = bert_ft
        results['BERT微调'].setdefault('train_time_sec', None)
        results['BERT微调'].setdefault('gpu_memory_MB', None)
        results['BERT微调'].setdefault('RE', None)

    # 兼容旧版文件名（如果你还保留着，也可以加载）
    if os.path.exists('bert_lr_results.pkl') and 'BERT+LR' not in results:
        results['BERT+LR'] = joblib.load('bert_lr_results.pkl')
    if os.path.exists('bert_finetune_results.pkl') and 'BERT微调' not in results:
        results['BERT微调'] = joblib.load('bert_finetune_results.pkl')

    return results

def generate_summary_report(results):
    print("\n" + "="*50)
    print("电商评论情感分析模型汇总报告")
    print("="*50)

    summary_data = []
    for model_name, metrics in results.items():
        row = {
            '模型': model_name,
            '准确率': metrics.get('accuracy'),
            '精确率': metrics.get('precision'),
            '召回率': metrics.get('recall'),
            'F1分数': metrics.get('f1'),
            '负面评论召回率': metrics.get('negative_recall'),
            '训练时间(秒)': metrics.get('train_time_sec'),
            'GPU显存(MB)': metrics.get('gpu_memory_MB'),
            '资源效率比(RE)': metrics.get('RE')
        }
        # 格式化 RE 和显存，避免过长小数
        # 安全格式化函数
        def safe_format(value, fmt='.6f'):
            if value is None:
                return None
            if isinstance(value, (int, float, np.floating, np.integer)):
                return f"{value:{fmt}}"
            return str(value)  # 已经是字符串则直接返回
        
        row['资源效率比(RE)'] = safe_format(row['资源效率比(RE)'], '.6f')
        row['GPU显存(MB)'] = safe_format(row['GPU显存(MB)'], '.2f')
        row['训练时间(秒)'] = safe_format(row['训练时间(秒)'], '.2f')
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    print("\n模型性能汇总:")
    print(df_summary.to_markdown(index=False))
    df_summary.to_csv('model_performance_summary.csv', index=False)

    # 可视化部分保持不变，但注意雷达图只用了三个指标，不会受影响
    visualize_comparison(results, df_summary)
    return df_summary

def visualize_comparison(results, df_summary):
    # 提取用于可视化的数值指标，忽略 None
    acc_list = [results[m].get('accuracy') for m in df_summary['模型']]
    f1_list = [results[m].get('f1') for m in df_summary['模型']]
    neg_rec_list = [results[m].get('negative_recall') for m in df_summary['模型']]
    # 如果所有模型都有 RE，可额外画图，这里暂不增加

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'negative_recall']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数', '负面评论召回率']
    model_names = df_summary['模型'].tolist()

    # 1. 柱状图对比（动态 Y 轴范围）
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        values = [results[model].get(metric) for model in model_names]
        # 过滤掉 None 值（理论上都有）
        valid = [(m, v) for m, v in zip(model_names, values) if v is not None]
        if not valid:
            continue
        names, vals = zip(*valid)
        bars = plt.bar(names, vals, color=plt.cm.tab10.colors[:len(names)])
        plt.title(f'{metric_names[i]} 对比')
        plt.ylabel(metric_names[i])
        plt.xticks(rotation=45 if len(names) > 3 else 0)

        vmin = min(vals) * 0.95
        plt.ylim(max(0, vmin), 1.0)

        for j, v in enumerate(vals):
            plt.text(j, v + 0.005, f"{v:.4f}", ha='center', fontsize=8)
    plt.suptitle('模型性能综合对比', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存 model_comparison.png")

    # 2. 混淆矩阵对比
    if len(results) == 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        axes = [ax]
    else:
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(n*5, 4.5))
        if n == 1:
            axes = [axes]
    for i, model_name in enumerate(model_names):
        metrics = results[model_name]
        if 'confusion_matrix' not in metrics or metrics['confusion_matrix'] is None:
            continue
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
        axes[i].set_title(f'{model_name} 混淆矩阵')
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
    plt.suptitle('模型混淆矩阵对比', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存 confusion_matrix_comparison.png")

    # 3. 雷达图对比（只选核心指标）
    radar_metrics = ['accuracy', 'f1', 'negative_recall']
    radar_labels = ['准确率', 'F1分数', '负面评论召回率']
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model_name in model_names:
        m = results[model_name]
        if all(k in m and m[k] is not None for k in radar_metrics):
            values = [m['accuracy'], m['f1'], m['negative_recall']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('模型性能雷达图对比', size=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig('performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存 performance_radar.png")


if __name__ == "__main__":
    all_results = load_results()
    if not all_results:
        print("未找到任何模型结果文件！请先运行各个模型脚本。")
    else:
        report = generate_summary_report(all_results)
        print("\n汇总报告生成完成！请查看以下文件：")
        print("- model_performance_summary.csv")
        print("- model_comparison.png")
        print("- confusion_matrix_comparison.png")
        print("- performance_radar.png")