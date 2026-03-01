"""
Flask Backend for Customer Churn Prediction Dashboard
Serves API endpoints + premium HTML/CSS/JS frontend
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import shap
import io
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Load Models & Data on Startup ───────────────────────────────────────────
print("Loading models...")
models = {
    'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
    'Decision Tree': joblib.load('models/decision_tree.pkl'),
    'Random Forest': joblib.load('models/random_forest.pkl'),
    'XGBoost': joblib.load('models/xgboost.pkl')
}
encoders = joblib.load('models/label_encoders.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
evaluation_results = joblib.load('models/evaluation_results.pkl')
roc_data = joblib.load('models/roc_data.pkl')
test_data = joblib.load('models/test_data.pkl')
print("✓ All models loaded successfully!")

# Load dataset info globally so we don't reload on every request
try:
    dataset = pd.read_csv('data/customer_churn_dataset-training-master.csv')
    
    # Mock Customer Name
    first_names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]
    np.random.seed(42)
    dataset['Customer Name'] = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(len(dataset))]
    
    # Mock Churn Date for churned customers (last 365 days)
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365)
    
    if 'Churn' in dataset.columns:
        if dataset['Churn'].dtype == 'object':
            churn_mask = dataset['Churn'].str.lower().isin(['yes', '1', 'true'])
        else:
            churn_mask = dataset['Churn'] == 1
            
        num_churned = churn_mask.sum()
        random_days = np.random.randint(0, 365, size=num_churned)
        
        churn_dates = pd.Series([pd.NaT] * len(dataset))
        churn_dates.loc[churn_mask] = start_date + pd.to_timedelta(random_days, unit='D')
        dataset['Churn Date'] = churn_dates.dt.date

    print(f"✓ Dataset loaded: {len(dataset):,} records with mock demographics")
except FileNotFoundError:
    dataset = None
    print("⚠ Dataset not found for stats")


def preprocess_input(data):
    """Preprocess input data for prediction"""
    df = data.copy()
    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column].astype(str))
    df_scaled = pd.DataFrame(
        scaler.transform(df[feature_columns]),
        columns=feature_columns,
        index=df.index
    )
    return df_scaled


def get_risk_level(prob):
    if prob < 0.4:
        return "Low Risk"
    elif prob < 0.75:
        return "Medium Risk"
    else:
        return "High Risk"


def build_tree_viz(dt_model, feature_cols, path_set):
    """Build full tree structure for Plotly tree visualization"""
    tree_ = dt_model.tree_

    # Pass 1: Assign sequential x positions to leaves via in-order traversal
    leaf_positions = {}
    counter = [0]

    def assign_leaves(nid):
        left = int(tree_.children_left[nid])
        right = int(tree_.children_right[nid])
        if left == right:  # leaf
            leaf_positions[nid] = counter[0]
            counter[0] += 1
        else:
            assign_leaves(left)
            assign_leaves(right)

    assign_leaves(0)

    # Pass 2: Build nodes and edges bottom-up
    nodes = []
    edges = []  # [x0, y0, x1, y1, on_path]
    pos_x = {}
    pos_y = {}

    def build(nid, depth):
        left = int(tree_.children_left[nid])
        right = int(tree_.children_right[nid])
        is_leaf = (left == right)
        on_path = nid in path_set

        if is_leaf:
            x = leaf_positions[nid]
            counts = tree_.value[nid][0]
            pred = 1 if len(counts) > 1 and counts[1] > counts[0] else 0
            label = 'Churn' if pred == 1 else 'Stay'
        else:
            build(left, depth + 1)
            build(right, depth + 1)
            x = (pos_x[left] + pos_x[right]) / 2
            fi = tree_.feature[nid]
            feat = feature_cols[fi]
            thr = round(float(tree_.threshold[nid]), 2)
            label = f"{feat} ≤ {thr}"

            left_on = on_path and left in path_set
            right_on = on_path and right in path_set
            edges.append([float(x), float(-depth), float(pos_x[left]), float(pos_y[left]), left_on])
            edges.append([float(x), float(-depth), float(pos_x[right]), float(pos_y[right]), right_on])

        pos_x[nid] = x
        pos_y[nid] = -depth
        nodes.append({
            'id': nid, 'x': float(x), 'y': float(-depth),
            'label': label, 'is_leaf': is_leaf, 'on_path': on_path
        })

    build(0, 0)
    return nodes, edges


def get_shap_reasons(input_processed, raw_values):
    """Generate SHAP-based churn reasons for a single prediction"""
    rf_model = models['Random Forest']
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(input_processed)

    # Handle different SHAP output formats
    if isinstance(shap_vals, list):
        # List of arrays: one per class
        individual_shap = shap_vals[1][0]  # Class 1 (churn) SHAP values
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        # 3D array: (n_samples, n_features, n_classes)
        individual_shap = shap_vals[0, :, 1]  # Class 1 (churn) SHAP values
    else:
        individual_shap = shap_vals[0]

    feature_impacts = []
    for i, col in enumerate(feature_columns):
        feature_impacts.append({
            'feature': col,
            'shap_value': float(individual_shap[i]),
            'raw_value': str(raw_values.get(col, '')),
            'abs_impact': float(abs(individual_shap[i]))
        })
    feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)

    reason_templates = {
        'Tenure': {
            'churn': "Short Tenure ({val} months) — Newer customers are more likely to leave as they haven't built loyalty yet.",
            'stay': "Long Tenure ({val} months) — Long-term customer indicating strong brand loyalty."
        },
        'Support Calls': {
            'churn': "High Support Calls ({val}) — Frequent support calls suggest dissatisfaction with the service.",
            'stay': "Low Support Calls ({val}) — Few support calls indicate customer satisfaction."
        },
        'Payment Delay': {
            'churn': "Payment Delays ({val} days) — Late payments may indicate disengagement or billing issues.",
            'stay': "Timely Payments ({val} days delay) — Consistent payments show commitment."
        },
        'Usage Frequency': {
            'churn': "Low Usage ({val} times/month) — Low engagement increases churn risk.",
            'stay': "High Usage ({val} times/month) — Active usage shows the customer finds value."
        },
        'Total Spend': {
            'churn': "Low Total Spend (${val}) — Lower spending may indicate the customer is not invested.",
            'stay': "High Total Spend (${val}) — Higher spending indicates strong commitment."
        },
        'Contract Length': {
            'churn': "{val} Contract — Shorter contracts make it easier to leave.",
            'stay': "{val} Contract — Longer contracts provide stability and reduce churn."
        },
        'Last Interaction': {
            'churn': "Recent Interaction ({val} days ago) — Recent contact may indicate unresolved problems.",
            'stay': "Last Interaction ({val} days ago) — No recent issues reported."
        },
        'Subscription Type': {
            'churn': "{val} Plan — This plan tier shows higher churn rates.",
            'stay': "{val} Plan — This plan tier shows better retention."
        },
        'Age': {
            'churn': "Age ({val}) — This age group shows higher churn tendency historically.",
            'stay': "Age ({val}) — This age group tends to be more stable."
        },
        'Gender': {
            'churn': "Gender ({val}) — This demographic shows slightly higher churn.",
            'stay': "Gender ({val}) — This demographic shows stable retention."
        }
    }

    churn_reasons = []
    stay_reasons = []

    for impact in feature_impacts:
        feat = impact['feature']
        val = impact['raw_value']
        shap_val = impact['shap_value']
        if feat in reason_templates:
            if shap_val > 0.01:
                churn_reasons.append({
                    'feature': feat,
                    'reason': reason_templates[feat]['churn'].format(val=val),
                    'impact': round(shap_val, 4)
                })
            elif shap_val < -0.01:
                stay_reasons.append({
                    'feature': feat,
                    'reason': reason_templates[feat]['stay'].format(val=val),
                    'impact': round(abs(shap_val), 4)
                })

    # Actionable recommendations
    recommendations = []
    if any(r['feature'] == 'Support Calls' for r in churn_reasons):
        recommendations.append("Assign a dedicated account manager to address pending support issues proactively.")
    if any(r['feature'] == 'Payment Delay' for r in churn_reasons):
        recommendations.append("Offer flexible payment plans or auto-pay discounts to reduce payment friction.")
    if any(r['feature'] == 'Tenure' for r in churn_reasons):
        recommendations.append("Introduce onboarding perks and early loyalty rewards for newer customers.")
    if any(r['feature'] == 'Usage Frequency' for r in churn_reasons):
        recommendations.append("Launch a personalized engagement campaign with usage tips and feature highlights.")
    if any(r['feature'] == 'Contract Length' for r in churn_reasons):
        recommendations.append("Offer discounted annual plans with added benefits to encourage longer commitments.")
    if any(r['feature'] == 'Total Spend' for r in churn_reasons):
        recommendations.append("Demonstrate ROI and offer tailored upgrade packages to increase perceived value.")
    if not recommendations:
        recommendations = [
            "Schedule a personalized satisfaction check-in call.",
            "Offer a loyalty discount or exclusive perk.",
            "Propose a longer-term contract with added benefits."
        ]

    return {
        'churn_reasons': churn_reasons[:5],
        'stay_reasons': stay_reasons[:5],
        'feature_impacts': feature_impacts,
        'recommendations': recommendations
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/customer_history', methods=['GET'])
def customer_history():
    if dataset is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
        
    df = dataset.copy()
    
    # Query params
    search_query = request.args.get('search', '').lower()
    start_date_str = request.args.get('start', '')
    end_date_str = request.args.get('end', '')
    
    # Filter by Search Query
    if search_query:
        id_col = next((col for col in ['CustomerID', 'customerID', 'customer_id', 'ID', 'id'] if col in df.columns), None)
        search_mask = pd.Series(False, index=df.index)
        
        if id_col:
            search_mask = search_mask | df[id_col].astype(str).str.lower().str.contains(search_query, na=False)
        if 'Customer Name' in df.columns:
            search_mask = search_mask | df['Customer Name'].astype(str).str.lower().str.contains(search_query, na=False)
            
        df = df[search_mask]
    
    # Filter by Dates (only for churned customers)
    elif start_date_str and end_date_str:
        try:
            start_date = pd.to_datetime(start_date_str).date()
            end_date = pd.to_datetime(end_date_str).date()
            date_mask = (df['Churn Date'] >= start_date) & (df['Churn Date'] <= end_date)
            df = df[date_mask]
        except Exception as e:
            return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
    else:
        # Default: show all churned
        churn_mask = df['Churn Date'].notna()
        df = df[churn_mask]
        
    # Calculate metrics
    total_customers = len(df)
    churned_count = df['Churn Date'].notna().sum()
    
    spend_col = next((col for col in ['Total Spend', 'MonthlyCharges', 'TotalCharges'] if col in df.columns), None)
    total_spend = 0
    if spend_col:
        df[spend_col] = pd.to_numeric(df[spend_col], errors='coerce')
        total_spend = float(df[spend_col].sum())
        
    # Prepare data for frontend
    # Convert dates to string and replace NaNs with None for JSON serialization
    df['Churn Date'] = df['Churn Date'].astype(str)
    df = df.replace({np.nan: None, 'NaT': None})
    
    # Reorder columns to show important ones first
    display_cols = list(df.columns)
    priority_cols = []
    for col in ['CustomerID', 'customerID', 'Customer Name', 'Churn Date', spend_col, 'Tenure']:
        if col and col in display_cols:
            priority_cols.append(col)
            display_cols.remove(col)
            
    final_cols = priority_cols + display_cols
    
    return jsonify({
        'metrics': {
            'total_found': total_customers,
            'total_spend': total_spend,
            'churned_in_period': int(churned_count)
        },
        'columns': final_cols,
        # Only send top 1000 to prevent browser crash
        'data': df[final_cols].head(1000).to_dict(orient='records')
    })


@app.route('/api/dashboard')
def dashboard_stats():
    """Return dashboard KPIs and dataset stats"""
    if dataset is not None:
        total = len(dataset)
        if dataset['Churn'].dtype in ['float64', 'int64']:
            churned = int(dataset['Churn'].sum())
        else:
            churned = int(dataset['Churn'].value_counts().get('Yes', 0))
        churn_rate = round((churned / total) * 100, 2)
        active = total - churned
    else:
        total, churned, active, churn_rate = 0, 0, 0, 0

    # Model performance
    perf = {}
    for name, metrics in evaluation_results.items():
        perf[name] = {
            'accuracy': round(metrics['accuracy'] * 100, 2),
            'precision': round(metrics['precision'] * 100, 2),
            'recall': round(metrics['recall'] * 100, 2),
            'f1_score': round(metrics['f1_score'] * 100, 2),
            'auc': round(metrics['auc'] * 100, 2)
        }

    best = max(evaluation_results.items(), key=lambda x: x[1]['f1_score'])

    return jsonify({
        'total_customers': total,
        'churned': churned,
        'active': active,
        'churn_rate': churn_rate,
        'model_performance': perf,
        'best_model': best[0],
        'best_f1': round(best[1]['f1_score'] * 100, 2),
        'features': feature_columns
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Single customer prediction with churn reasons"""
    data = request.json

    raw_values = {
        'Age': data.get('Age', 35),
        'Gender': data.get('Gender', 'Male'),
        'Tenure': data.get('Tenure', 12),
        'Usage Frequency': data.get('Usage Frequency', 15),
        'Support Calls': data.get('Support Calls', 2),
        'Payment Delay': data.get('Payment Delay', 5),
        'Subscription Type': data.get('Subscription Type', 'Basic'),
        'Contract Length': data.get('Contract Length', 'Monthly'),
        'Total Spend': data.get('Total Spend', 500),
        'Last Interaction': data.get('Last Interaction', 10)
    }

    input_df = pd.DataFrame({k: [v] for k, v in raw_values.items()})
    input_processed = preprocess_input(input_df)

    # All model predictions
    results = {}
    for name, model in models.items():
        pred = int(model.predict(input_processed)[0])
        prob = float(model.predict_proba(input_processed)[0][1])
        results[name] = {'prediction': pred, 'probability': round(prob, 4)}

    # Primary model (Random Forest)
    rf = models['Random Forest']
    prediction = int(rf.predict(input_processed)[0])
    probability = float(rf.predict_proba(input_processed)[0][1])
    risk_level = get_risk_level(probability)

    # Feature importance
    fi = rf.feature_importances_
    feature_importance = [
        {'feature': feature_columns[i], 'importance': round(float(fi[i]), 4)}
        for i in range(len(feature_columns))
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)

    # SHAP reasons
    try:
        reasons = get_shap_reasons(input_processed, raw_values)
    except Exception as e:
        reasons = {'churn_reasons': [], 'stay_reasons': [], 'feature_impacts': [], 'recommendations': [str(e)]}

    return jsonify({
        'prediction': prediction,
        'probability': round(probability, 4),
        'risk_level': risk_level,
        'all_models': results,
        'feature_importance': feature_importance,
        'reasons': reasons,
        'customer_data': raw_values
    })


@app.route('/api/bulk-predict', methods=['POST'])
def bulk_predict():
    """Bulk prediction from CSV"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        bulk_data = pd.read_csv(file)
        original_data = bulk_data.copy()

        # Remove ID columns
        for id_col in ['CustomerID', 'customerID', 'customer_id', 'ID', 'id']:
            if id_col in bulk_data.columns:
                bulk_data = bulk_data.drop(id_col, axis=1)
                break

        # Remove Churn if present
        if 'Churn' in bulk_data.columns:
            bulk_data = bulk_data.drop('Churn', axis=1)

        processed = preprocess_input(bulk_data)
        rf = models['Random Forest']
        predictions = rf.predict(processed)
        probabilities = rf.predict_proba(processed)[:, 1]

        results = []
        for i in range(len(original_data)):
            row = original_data.iloc[i].to_dict()
            # Convert numpy types to Python types for JSON
            for k, v in row.items():
                if isinstance(v, (np.integer,)):
                    row[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    row[k] = float(v)

            results.append({
                'index': i + 1,
                'data': row,
                'prediction': int(predictions[i]),
                'probability': round(float(probabilities[i]), 4),
                'risk_level': get_risk_level(probabilities[i])
            })

        total = len(results)
        churned = int(predictions.sum())
        high_risk = sum(1 for r in results if r['risk_level'] == 'High Risk')
        medium_risk = sum(1 for r in results if r['risk_level'] == 'Medium Risk')
        low_risk = sum(1 for r in results if r['risk_level'] == 'Low Risk')

        return jsonify({
            'success': True,
            'total': total,
            'churned': churned,
            'churn_rate': round((churned / total) * 100, 2) if total > 0 else 0,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'avg_probability': round(float(probabilities.mean()), 4),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-analytics')
def model_analytics():
    """Return model analytics data"""
    analytics = {}
    for name, metrics in evaluation_results.items():
        analytics[name] = {
            'accuracy': round(metrics['accuracy'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'f1_score': round(metrics['f1_score'], 4),
            'auc': round(metrics['auc'], 4),
            'confusion_matrix': metrics['confusion_matrix']
        }

    # ROC data
    roc = {}
    for name, data in roc_data.items():
        # Downsample ROC points for frontend performance
        fpr = data['fpr']
        tpr = data['tpr']
        if len(fpr) > 200:
            step = len(fpr) // 200
            fpr = fpr[::step]
            tpr = tpr[::step]
        roc[name] = {
            'fpr': [round(float(x), 4) for x in fpr],
            'tpr': [round(float(x), 4) for x in tpr],
            'auc': round(float(data['auc']), 4)
        }

    return jsonify({
        'analytics': analytics,
        'roc': roc
    })


@app.route('/api/explainability')
def explainability():
    """Return SHAP feature importance"""
    X_test = test_data['X_test']
    sample = X_test.iloc[:min(500, len(X_test))]

    result = {}
    for model_name in ['Random Forest', 'XGBoost']:
        model = models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(sample)

        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            sv = shap_vals[:, :, 1]  # Class 1 (churn) SHAP values
        else:
            sv = shap_vals

        mean_abs = np.abs(sv).mean(axis=0)
        importance = [
            {'feature': feature_columns[i], 'importance': round(float(mean_abs[i]), 4)}
            for i in range(len(feature_columns))
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)
        result[model_name] = importance

    return jsonify(result)


@app.route('/api/decision-tree', methods=['POST'])
def decision_tree_predict():
    """Decision Tree prediction with decision path visualization"""
    data = request.json

    raw_values = {
        'Age': data.get('Age', 35),
        'Gender': data.get('Gender', 'Male'),
        'Tenure': data.get('Tenure', 12),
        'Usage Frequency': data.get('Usage Frequency', 15),
        'Support Calls': data.get('Support Calls', 2),
        'Payment Delay': data.get('Payment Delay', 5),
        'Subscription Type': data.get('Subscription Type', 'Basic'),
        'Contract Length': data.get('Contract Length', 'Monthly'),
        'Total Spend': data.get('Total Spend', 500),
        'Last Interaction': data.get('Last Interaction', 10)
    }

    input_df = pd.DataFrame({k: [v] for k, v in raw_values.items()})
    input_processed = preprocess_input(input_df)

    dt_model = models['Decision Tree']
    prediction = int(dt_model.predict(input_processed)[0])
    probability = float(dt_model.predict_proba(input_processed)[0][1])
    risk_level = get_risk_level(probability)

    # Extract decision path
    tree = dt_model.tree_
    node_indicator = dt_model.decision_path(input_processed)
    node_indices = node_indicator.indices

    decision_path = []
    for i, node_id in enumerate(node_indices):
        is_leaf = (tree.children_left[node_id] == tree.children_right[node_id])

        if is_leaf:
            # Leaf node — get class distribution
            class_counts = tree.value[node_id][0]
            total = class_counts.sum()
            churn_pct = round((class_counts[1] / total) * 100, 1) if total > 0 else 0
            stay_pct = round((class_counts[0] / total) * 100, 1) if total > 0 else 0
            decision_path.append({
                'type': 'leaf',
                'prediction': 'Churn' if prediction == 1 else 'Stay',
                'churn_pct': churn_pct,
                'stay_pct': stay_pct,
                'samples': int(total)
            })
        else:
            feat_idx = tree.feature[node_id]
            threshold = round(float(tree.threshold[node_id]), 2)
            feat_name = feature_columns[feat_idx]
            sample_val = float(input_processed.iloc[0, feat_idx])
            went_left = sample_val <= threshold

            decision_path.append({
                'type': 'split',
                'feature': feat_name,
                'threshold': threshold,
                'direction': 'left' if went_left else 'right',
                'condition': f"{feat_name} <= {threshold}",
                'result': 'Yes' if went_left else 'No',
                'value': round(sample_val, 2)
            })

    # Feature importance from DT
    fi = dt_model.feature_importances_
    feature_importance = [
        {'feature': feature_columns[i], 'importance': round(float(fi[i]), 4)}
        for i in range(len(feature_columns))
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)

    # Build full tree visualization
    path_set = set(int(n) for n in node_indices)
    tree_nodes, tree_edges = build_tree_viz(dt_model, feature_columns, path_set)

    return jsonify({
        'prediction': prediction,
        'probability': round(probability, 4),
        'risk_level': risk_level,
        'decision_path': decision_path,
        'feature_importance': feature_importance,
        'customer_data': raw_values,
        'tree_depth': int(dt_model.get_depth()),
        'tree_leaves': int(dt_model.get_n_leaves()),
        'tree_viz': {
            'nodes': tree_nodes,
            'edges': tree_edges
        }
    })


@app.route('/api/monthly-churn', methods=['POST'])
def monthly_churn():
    """Month-wise churn analysis using Tenure to simulate join dates"""
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    data = request.json or {}
    from_month = data.get('from_month')  # "2024-01" format
    to_month = data.get('to_month')      # "2026-02" format

    df_copy = pd.read_csv('data/customer_churn_dataset-training-master.csv')
    df_copy = df_copy.dropna(subset=['Tenure', 'Churn'])

    # Assign a "join month" based on tenure (months ago from now)
    now = datetime.now()
    df_copy['join_date'] = df_copy['Tenure'].apply(
        lambda t: now - relativedelta(months=int(t))
    )
    df_copy['month'] = df_copy['join_date'].dt.to_period('M').astype(str)

    # Filter by date range if specified
    all_months = sorted(df_copy['month'].unique())
    if from_month:
        df_copy = df_copy[df_copy['month'] >= from_month]
    if to_month:
        df_copy = df_copy[df_copy['month'] <= to_month]

    # Group by month and churn status
    monthly = df_copy.groupby('month')['Churn'].agg(['sum', 'count']).reset_index()
    monthly.columns = ['month', 'churned', 'total']
    monthly['stayed'] = monthly['total'] - monthly['churned']
    monthly['churn_rate'] = (monthly['churned'] / monthly['total'] * 100).round(1)
    monthly = monthly.sort_values('month')

    return jsonify({
        'months': monthly['month'].tolist(),
        'churned': monthly['churned'].astype(int).tolist(),
        'stayed': monthly['stayed'].astype(int).tolist(),
        'total': monthly['total'].astype(int).tolist(),
        'churn_rate': monthly['churn_rate'].tolist(),
        'available_months': all_months,
        'summary': {
            'total_customers': int(monthly['total'].sum()),
            'total_churned': int(monthly['churned'].sum()),
            'total_stayed': int(monthly['stayed'].sum()),
            'avg_churn_rate': round(float(monthly['churn_rate'].mean()), 1)
        }
    })


@app.route('/api/download-csv', methods=['POST'])
def download_csv():
    """Download bulk prediction results as CSV"""
    data = request.json
    if not data or 'results' not in data:
        return jsonify({'error': 'No data'}), 400

    rows = []
    for r in data['results']:
        row = r.get('data', {})
        row['Predicted_Churn'] = r['prediction']
        row['Churn_Probability'] = r['probability']
        row['Risk_Level'] = r['risk_level']
        rows.append(row)

    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='churn_predictions.csv'
    )


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🎯 Smart Churn Predictor — Flask Server")
    print("  📍 Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
