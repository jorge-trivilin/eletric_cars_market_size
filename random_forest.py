from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')

data = pd.read_csv(
    'data/Electric_Vehicle_Population_Data.csv'
)

X_no_range = data[['Model Year', 'Base MSRP']]  
y = data['Electric Vehicle Type']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss'}

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results_no_range = cross_validate(
    pipeline, 
    X_no_range, 
    y_encoded, 
    cv=cv, 
    scoring=scoring,
    return_train_score=True
)

X_all = data[['Model Year', 'Electric Range', 'Base MSRP']]
cv_results_all = cross_validate(
    pipeline, 
    X_all, 
    y_encoded, 
    cv=cv, 
    scoring=scoring,
    return_train_score=True
)

print("Random Forest - Cross-validation results without Electric Range:")
print(f"Mean accuracy: {cv_results_no_range['test_accuracy'].mean():.4f} ± {cv_results_no_range['test_accuracy'].std():.4f}")
print(f"Mean log loss: {-cv_results_no_range['test_log_loss'].mean():.4f} ± {cv_results_no_range['test_log_loss'].std():.4f}")
print("Accuracy by fold:", cv_results_no_range['test_accuracy'])
print("Log loss by fold:", -cv_results_no_range['test_log_loss'])

print("\nRandom Forest - Cross-validation results with all features:")
print(f"Mean accuracy: {cv_results_all['test_accuracy'].mean():.4f} ± {cv_results_all['test_accuracy'].std():.4f}")
print(f"Mean log loss: {-cv_results_all['test_log_loss'].mean():.4f} ± {cv_results_all['test_log_loss'].std():.4f}")
print("Accuracy by fold:", cv_results_all['test_accuracy'])
print("Log loss by fold:", -cv_results_all['test_log_loss'])

feature_importances = []
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all_processed = pd.get_dummies(X_all, drop_first=True)
fold_metrics = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all_processed, y_encoded)):
    X_train_fold, X_test_fold = X_all_processed.iloc[train_idx], X_all_processed.iloc[test_idx]
    y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]
    
    fold_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fold_model.fit(X_train_fold, y_train_fold)
    
    y_pred_proba = fold_model.predict_proba(X_test_fold)
    fold_log_loss = log_loss(y_test_fold, y_pred_proba)
    
    fold_metrics.append({
        'fold': fold_idx + 1,
        'log_loss': fold_log_loss
    })
    
    fold_importance = pd.DataFrame({
        'Feature': X_all_processed.columns,
        'Importance': fold_model.feature_importances_,
        'Fold': fold_idx + 1
    })
    feature_importances.append(fold_importance)

print("\nDetailed log loss by fold:")
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df)

avg_importance = pd.concat(feature_importances).groupby('Feature').mean().sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y=avg_importance.index, data=avg_importance.reset_index())
plt.title('Random Forest - Average Feature Importance Across CV Folds')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('rf_feature_importances.png')

avg_importance['Model'] = 'Random Forest'

xgb_feature_importances = []
for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all_processed, y_encoded)):
    X_train_fold, y_train_fold = X_all_processed.iloc[train_idx], y_encoded[train_idx]
    
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train_fold, y_train_fold)
    
    xgb_fold_importance = pd.DataFrame({
        'Feature': X_all_processed.columns,
        'Importance': xgb_model.feature_importances_,
        'Fold': fold_idx + 1
    })
    xgb_feature_importances.append(xgb_fold_importance)

xgb_avg_importance = pd.concat(xgb_feature_importances).groupby('Feature').mean().sort_values('Importance', ascending=False)
xgb_avg_importance['Model'] = 'XGBoost'

combined_importance = pd.concat([
    avg_importance.reset_index(), 
    xgb_avg_importance.reset_index()
])

plt.figure(figsize=(14, 8))
sns.barplot(x='Importance', y='Feature', hue='Model', data=combined_importance)
plt.title('Feature Importance: Random Forest vs XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('model_comparison_feature_importances.png')

train_test_comparison = pd.DataFrame({
    'RF Train Accuracy': cv_results_all['train_accuracy'].mean(),
    'RF Test Accuracy': cv_results_all['test_accuracy'].mean(),
    'RF Train-Test Gap': cv_results_all['train_accuracy'].mean() - cv_results_all['test_accuracy'].mean()
}, index=[0])

print("\nRandom Forest - Train vs Test Performance:")
print(train_test_comparison)