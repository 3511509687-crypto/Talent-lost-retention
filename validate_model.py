import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# 设置随机种子以确保可复现性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 数据路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(CURRENT_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 加载数据
def load_data():
    df = pd.read_csv(data_path)
    return df

# 数据预处理
def preprocess_data(df):
    # 处理目标变量
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # 选择特征
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
                     'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
                     'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
                     'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
                     'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
                     'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    return X, y, preprocessor

# 计算类别权重
def compute_scale_pos_weight(y):
    class_counts = np.bincount(y)
    scale_pos_weight = class_counts[0] / class_counts[1]
    return scale_pos_weight

# 构建模型
def build_models(scale_pos_weight):
    # LightGBM参数（使用修改后的参数）
    lgb_params = {
        'num_leaves': 15,
        'learning_rate': 0.03,
        'n_estimators': 150,
        'max_depth': 3,
        'min_child_samples': 60,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 2.0,
        'reg_lambda': 4.0,
        'min_split_gain': 0.2,
        'objective': 'binary',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight
    }
    
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    
    lr_clf = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced',
        C=0.35,
        random_state=RANDOM_STATE
    )
    
    et_clf = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    return lgb_clf, lr_clf, et_clf

# 模型融合类
class BlendedModel:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        probs = []
        for model in self.models.values():
            prob = model.predict_proba(X)[:, 1]
            probs.append(prob)
        # 简单平均融合
        avg_prob = np.mean(probs, axis=0)
        return np.column_stack([1 - avg_prob, avg_prob])
    
    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)

# 验证模型
def validate_model():
    print("=== 验证模型训练 ===")
    
    # 加载数据
    df = load_data()
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 数据预处理
    X, y, preprocessor = preprocess_data(df)
    print(f"特征数量: {X.shape[1]}")
    print(f"目标变量分布: {np.bincount(y)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 预处理数据
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"预处理后训练集形状: {X_train_trans.shape}")
    print(f"预处理后测试集形状: {X_test_trans.shape}")
    
    # 计算类别权重
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"类别权重: {scale_pos_weight}")
    
    # 构建模型
    lgb_clf, lr_clf, et_clf = build_models(scale_pos_weight)
    
    # 创建融合模型
    model_map = {
        'lgb': lgb_clf,
        'lr': lr_clf,
        'et': et_clf
    }
    blended_model = BlendedModel(model_map)
    
    # 训练模型
    print("开始训练模型...")
    blended_model.fit(X_train_trans, y_train)
    print("模型训练完成")
    
    # 预测
    y_train_prob = blended_model.predict_proba(X_train_trans)[:, 1]
    y_test_prob = blended_model.predict_proba(X_test_trans)[:, 1]
    
    # 计算阈值
    threshold = 0.5
    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_test_pred = (y_test_prob >= threshold).astype(int)
    
    # 计算评估指标
    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # 输出结果
    print("\n=== 模型评估结果 ===")
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"训练集 准确率: {train_acc:.4f}")
    print(f"测试集 准确率: {test_acc:.4f}")
    print(f"训练集 F1值: {train_f1:.4f}")
    print(f"测试集 F1值: {test_f1:.4f}")
    
    # 与原模型结果比较
    print("\n=== 与原模型结果比较 ===")
    print("原模型结果:")
    print("训练集 AUC: 0.915, 测试集 AUC: 0.835")
    print("训练集 准确率: 0.898, 测试集 准确率: 0.857")
    print("训练集 F1值: 0.694, 测试集 F1值: 0.553")
    
    print("\n验证模型结果:")
    print(f"训练集 AUC: {train_auc:.3f}, 测试集 AUC: {test_auc:.3f}")
    print(f"训练集 准确率: {train_acc:.3f}, 测试集 准确率: {test_acc:.3f}")
    print(f"训练集 F1值: {train_f1:.3f}, 测试集 F1值: {test_f1:.3f}")
    
    # 检查结果是否一致
    print("\n=== 结果一致性检查 ===")
    auc_diff = abs(test_auc - 0.835)
    acc_diff = abs(test_acc - 0.857)
    f1_diff = abs(test_f1 - 0.553)
    
    print(f"AUC差异: {auc_diff:.4f}")
    print(f"准确率差异: {acc_diff:.4f}")
    print(f"F1值差异: {f1_diff:.4f}")
    
    if auc_diff < 0.05 and acc_diff < 0.05 and f1_diff < 0.05:
        print("✅ 验证通过：模型训练结果可复现")
    else:
        print("❌ 验证失败：模型训练结果与原结果差异较大")

if __name__ == "__main__":
    validate_model()
