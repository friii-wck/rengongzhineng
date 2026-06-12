# -*- coding: utf-8 -*-
"""
亚马逊商品评论情感分析 - 课程设计增强版（满足报告图表数量要求）
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 自定义英文停用词表
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'a', 'an', 'and', 'or', 'but', 'if',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'without', 'after',
    'before', 'up', 'down', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must', 'this', 'that', 'these', 'those', 'such', 'each', 'every', 'all',
    'both', 'neither', 'either', 'some', 'any', 'no', 'none', 'very', 'just', 'not', 'too',
    'very', 'can', 'cannot', 'could', 'would', 'should', 'might', 'must', 'also', 'even',
    'like', 'so', 'than', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'until', 'upon', 'with', 'within', 'without', 'would', 'could', 'should', 'might', 'must'
])

def clean_and_tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return tokens

def process_text(text):
    tokens = clean_and_tokenize(text)
    return ' '.join(tokens)

# ==================== 1. 加载数据 ====================
print("="*60)
print("步骤1：加载数据")
print("="*60)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
possible_names = ['Reviews.csv', 'Reviews', 'reviews.csv', 'reviews']
data_file = None
for name in possible_names:
    test_path = os.path.join(parent_dir, name)
    if os.path.isfile(test_path):
        data_file = test_path
        break
if data_file is None:
    print("错误：未找到数据文件。请确保 Reviews.csv 在上级目录。")
    sys.exit(1)

print(f"找到数据文件：{data_file}")
df = pd.read_csv(data_file, low_memory=False)
print(f"原始数据总量：{len(df)} 条评论")

# ==================== 2. 生成情感标签并过滤中性 ====================
print("\n" + "="*60)
print("步骤2：生成情感标签")
print("="*60)
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
df_filtered = df[df['Sentiment'] != 'neutral'].copy()
print(f"过滤中性后剩余：{len(df_filtered)} 条")
print(f"  正面：{(df_filtered['Sentiment']=='positive').sum()}, 负面：{(df_filtered['Sentiment']=='negative').sum()}")

# ==================== 3. 分层采样 ====================
sample_size = 5000
print(f"\n步骤3：分层采样 {sample_size} 条")
df_sample, _ = train_test_split(df_filtered, train_size=sample_size,
                                stratify=df_filtered['Sentiment'], random_state=42)
print(f"采样后：正面{(df_sample['Sentiment']=='positive').sum()}, 负面{(df_sample['Sentiment']=='negative').sum()}")

# ==================== 4. 文本预处理 ====================
print("\n步骤4：文本预处理")
df_sample['Processed_Text'] = df_sample['Text'].apply(process_text)
print("示例：")
print(f"  原始：{df_sample['Text'].iloc[0][:100]}...")
print(f"  处理后：{df_sample['Processed_Text'].iloc[0][:100]}...")

# ==================== 5. TF-IDF特征提取 ====================
print("\n步骤5：TF-IDF特征提取")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8, stop_words='english')
X = tfidf.fit_transform(df_sample['Processed_Text'])
y = df_sample['Sentiment'].map({'positive': 1, 'negative': 0})
print(f"特征维度：{X.shape}")

# ==================== 6. 划分训练集和测试集 ====================
print("\n步骤6：划分训练集(80%)和测试集(20%)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集：{X_train.shape[0]}条 (正面{(y_train==1).sum()}, 负面{(y_train==0).sum()})")
print(f"测试集：{X_test.shape[0]}条 (正面{(y_test==1).sum()}, 负面{(y_test==0).sum()})")

# ==================== 7. 训练模型（使用最佳C）并绘制训练过程图 ====================
print("\n步骤7：模型训练与训练过程可视化")

# 7.1 使用验证曲线选择最佳C值，同时绘制训练过程图1：不同C下的准确率曲线
param_range = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
train_scores, test_scores = validation_curve(
    LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear', random_state=42),
    X_train, y_train, param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.semilogx(param_range, train_scores_mean, label='训练集准确率', marker='o')
plt.semilogx(param_range, test_scores_mean, label='交叉验证准确率', marker='s')
plt.xlabel('正则化强度参数 C')
plt.ylabel('准确率')
plt.title('图1：不同C值下的模型性能对比')
plt.legend()
plt.grid(True)
plt.savefig('training_curve_C_validation.png', dpi=150)
plt.close()
print("  已保存：training_curve_C_validation.png (训练过程图1)")

# 选择最佳C
best_idx = np.argmax(test_scores_mean)
best_C = param_range[best_idx]
print(f"  最佳C值：{best_C} (验证准确率：{test_scores_mean[best_idx]:.4f})")

# 使用最佳C重新训练模型（用于后续评估）
model = LogisticRegression(C=best_C, class_weight='balanced', max_iter=1000,
                           random_state=42, solver='liblinear', verbose=1)
model.fit(X_train, y_train)
print(f"\n模型训练完成，实际迭代次数：{model.n_iter_[0]}")

# 7.2 特征系数条形图（训练过程图2）
feature_names = tfidf.get_feature_names_out()
coef = model.coef_[0]
top_pos = np.argsort(coef)[-10:][::-1]
top_neg = np.argsort(coef)[:10]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.barh([feature_names[i] for i in top_pos][::-1], coef[top_pos][::-1], color='green')
plt.xlabel('系数值')
plt.title('正面影响最大的10个词')
plt.subplot(1,2,2)
plt.barh([feature_names[i] for i in top_neg][::-1], coef[top_neg][::-1], color='red')
plt.xlabel('系数值')
plt.title('负面影响最大的10个词')
plt.tight_layout()
plt.savefig('training_feature_coefficients.png', dpi=150)
plt.close()
print("  已保存：training_feature_coefficients.png (训练过程图2)")

# 7.3 绘制训练过程中的损失下降曲线（通过model.n_iter_无法得到序列，但可以用不同迭代次数模拟）
# 这里使用逐步增加迭代次数的方法来模拟损失下降（仅作示意，实际逻辑回归有闭式解）
# 更科学的方法：使用 SGDClassifier 的 loss_curve_，但为了简单，我们展示不同C下的损失变化即可。
# 作为替代，我们绘制训练集和测试集上随 C 值变化的损失（负对数似然）
from sklearn.metrics import log_loss
train_losses = []
test_losses = []
for c in param_range:
    lr = LogisticRegression(C=c, class_weight='balanced', max_iter=1000, solver='liblinear', random_state=42)
    lr.fit(X_train, y_train)
    y_train_prob = lr.predict_proba(X_train)
    y_test_prob = lr.predict_proba(X_test)
    train_losses.append(log_loss(y_train, y_train_prob))
    test_losses.append(log_loss(y_test, y_test_prob))

plt.figure(figsize=(8,6))
plt.semilogx(param_range, train_losses, label='训练集损失', marker='o')
plt.semilogx(param_range, test_losses, label='测试集损失', marker='s')
plt.xlabel('正则化强度参数 C')
plt.ylabel('对数损失 (Log Loss)')
plt.title('图3：不同C值下的损失变化')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_curve.png', dpi=150)
plt.close()
print("  已保存：training_loss_curve.png (训练过程图3)")

# ==================== 8. 测试集评估与测试结果图 ====================
print("\n" + "="*60)
print("步骤8：测试集评估及结果可视化")
print("="*60)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 正类概率

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

# 测试结果图1：混淆矩阵热力图
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
plt.title('图4：测试集混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('test_confusion_matrix.png', dpi=150)
plt.close()
print("  已保存：test_confusion_matrix.png (测试结果图1)")

# 测试结果图2：ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0,1], [0,1], 'k--', label='随机分类器')
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真阳性率 (True Positive Rate)')
plt.title('图5：ROC曲线')
plt.legend()
plt.grid(True)
plt.savefig('test_roc_curve.png', dpi=150)
plt.close()
print("  已保存：test_roc_curve.png (测试结果图2)")

# 测试结果图3：精确率-召回率曲线
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(recall_vals, precision_vals, linewidth=2, color='darkorange')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('图6：精确率-召回率曲线')
plt.grid(True)
plt.savefig('test_precision_recall_curve.png', dpi=150)
plt.close()
print("  已保存：test_precision_recall_curve.png (测试结果图3)")

# 额外的测试结果图4：预测置信度分布（可选，超过3张）
plt.figure(figsize=(8,5))
plt.hist(y_prob[y_test==1], bins=30, alpha=0.7, label='正面评论', color='green')
plt.hist(y_prob[y_test==0], bins=30, alpha=0.7, label='负面评论', color='red')
plt.xlabel('预测为正类的概率')
plt.ylabel('频数')
plt.title('图7：预测置信度分布（按真实类别区分）')
plt.legend()
plt.savefig('test_confidence_distribution.png', dpi=150)
plt.close()
print("  已保存：test_confidence_distribution.png (测试结果图4，可选)")

# ==================== 9. 其他可视化（情感分布、词云） ====================
print("\n步骤9：生成辅助可视化（情感分布、词云）")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sent_counts = df_sample['Sentiment'].value_counts()
axes[0,0].bar(sent_counts.index, sent_counts.values, color=['#66b3ff','#ff9999'])
axes[0,0].set_title('采样后情感分布')
axes[0,0].set_xlabel('情感')
axes[0,0].set_ylabel('数量')
for i,v in enumerate(sent_counts.values):
    axes[0,0].text(i, v+20, str(v), ha='center')

# 复用混淆矩阵热力图（也可以再生成一次，但这里留空避免重复，或者显示训练集混淆）
axes[0,1].axis('off')

pos_text = ' '.join(df_sample[df_sample['Sentiment']=='positive']['Processed_Text'])
if pos_text:
    wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(pos_text)
    axes[1,0].imshow(wc_pos, interpolation='bilinear')
    axes[1,0].axis('off')
    axes[1,0].set_title('正面评论词云')

neg_text = ' '.join(df_sample[df_sample['Sentiment']=='negative']['Processed_Text'])
if neg_text:
    wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(neg_text)
    axes[1,1].imshow(wc_neg, interpolation='bilinear')
    axes[1,1].axis('off')
    axes[1,1].set_title('负面评论词云')

plt.tight_layout()
plt.savefig('sentiment_wordclouds.png', dpi=150)
plt.show()
print("  已保存：sentiment_wordclouds.png")

# ==================== 10. 自定义评论测试 ====================
print("\n" + "="*60)
print("步骤10：自定义评论测试")
print("="*60)
test_reviews = [
    "This product is absolutely amazing! Great quality and fast delivery.",
    "Terrible product, broke after one week. Very disappointed.",
    "It's okay, nothing special but it works fine.",
    "Excellent value for money, highly recommended!",
    "Waste of money, do not buy this product."
]
for rev in test_reviews:
    proc = process_text(rev)
    vec = tfidf.transform([proc])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    label = "正面" if pred == 1 else "负面"
    print(f"评论: {rev[:60]}...")
    print(f"预测: {label} (置信度: {prob:.2%})\n")

print("="*60)
print("程序运行完毕，所有图表已保存至当前目录")
print("="*60)

# ==================== 13. 输出典型评论样例（供截图） ====================
print("\n" + "="*60)
print("典型评论样例（供实验报告截图使用）")
print("="*60)

# 设置随机种子保证每次选取相同样例
np.random.seed(42)

# 获取训练集和测试集对应的原始DataFrame索引及文本
# 注意：X_train, X_test 是稀疏矩阵，需要回溯到原始数据框
# 方法：训练集和测试集划分时保持了索引，我们可以用 y_train.index 获取原始df中的行号
train_indices = y_train.index
test_indices = y_test.index

# 获取原始文本、情感标签、预测结果和置信度
def get_sample_info(indices, X_sparse, y_series, model, tfidf, df_original):
    """
    根据索引列表，返回原始文本、真实标签、预测标签、置信度
    """
    results = []
    for idx in indices:
        text = df_original.loc[idx, 'Text']
        true_label = df_original.loc[idx, 'Sentiment']
        # 预测
        processed = df_original.loc[idx, 'Processed_Text']  # 已经有预处理后的文本
        vec = tfidf.transform([processed])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred]
        pred_label = "positive" if pred == 1 else "negative"
        results.append({
            'text': text,
            'true': true_label,
            'pred': pred_label,
            'conf': prob
        })
    return results

# 从训练集中分别选取正面和负面样例
train_pos_indices = [i for i in train_indices if y_train[i] == 1]
train_neg_indices = [i for i in train_indices if y_train[i] == 0]
# 各取5个（如果不足5个则取全部）
sample_pos_train = np.random.choice(train_pos_indices, size=min(5, len(train_pos_indices)), replace=False)
sample_neg_train = np.random.choice(train_neg_indices, size=min(5, len(train_neg_indices)), replace=False)

# 从测试集中分别选取正面和负面样例
test_pos_indices = [i for i in test_indices if y_test[i] == 1]
test_neg_indices = [i for i in test_indices if y_test[i] == 0]
sample_pos_test = np.random.choice(test_pos_indices, size=min(5, len(test_pos_indices)), replace=False)
sample_neg_test = np.random.choice(test_neg_indices, size=min(5, len(test_neg_indices)), replace=False)

# 输出训练集样例
print("\n【训练集 - 正面评论示例】")
for i, idx in enumerate(sample_pos_train, 1):
    info = get_sample_info([idx], None, None, model, tfidf, df_sample)[0]  # 复用函数
    print(f"{i}. 真实:正面 | 预测:{info['pred']} | 置信度:{info['conf']:.2%}")
    print(f"   评论: {info['text'][:150]}...")
    print()

print("\n【训练集 - 负面评论示例】")
for i, idx in enumerate(sample_neg_train, 1):
    info = get_sample_info([idx], None, None, model, tfidf, df_sample)[0]
    print(f"{i}. 真实:负面 | 预测:{info['pred']} | 置信度:{info['conf']:.2%}")
    print(f"   评论: {info['text'][:150]}...")
    print()

# 输出测试集样例
print("\n【测试集 - 正面评论示例】")
for i, idx in enumerate(sample_pos_test, 1):
    info = get_sample_info([idx], None, None, model, tfidf, df_sample)[0]
    print(f"{i}. 真实:正面 | 预测:{info['pred']} | 置信度:{info['conf']:.2%}")
    print(f"   评论: {info['text'][:150]}...")
    print()

print("\n【测试集 - 负面评论示例】")
for i, idx in enumerate(sample_neg_test, 1):
    info = get_sample_info([idx], None, None, model, tfidf, df_sample)[0]
    print(f"{i}. 真实:负面 | 预测:{info['pred']} | 置信度:{info['conf']:.2%}")
    print(f"   评论: {info['text'][:150]}...")
    print()

print("="*60)
print("典型评论样例输出完毕，可直接截图用于实验报告")
print("="*60)