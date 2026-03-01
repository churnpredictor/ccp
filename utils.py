"""
Utility functions for Customer Churn Prediction Dashboard
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io


def preprocess_input(data, feature_encoders, scaler, feature_columns):
    """
    Preprocess input data for prediction
    
    Args:
        data: DataFrame with customer data
        feature_encoders: Dictionary of label encoders
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
    
    Returns:
        Preprocessed data ready for model prediction
    """
    df = data.copy()
    
    # Encode categorical features
    for column, encoder in feature_encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column].astype(str))
    
    # Scale numerical features
    df_scaled = pd.DataFrame(
        scaler.transform(df[feature_columns]),
        columns=feature_columns,
        index=df.index
    )
    
    return df_scaled


def create_probability_gauge(probability, risk_level):
    """
    Create a Plotly gauge chart for churn probability
    
    Args:
        probability: Churn probability (0-1)
        risk_level: Risk level string ("Low", "Medium", "High")
    
    Returns:
        Plotly figure object
    """
    # Determine color based on risk level
    if risk_level == "Low Risk":
        color = "#10B981"  # Green
    elif risk_level == "Medium Risk":
        color = "#F59E0B"  # Orange
    else:
        color = "#EF4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Churn Probability</b><br><span style='font-size:0.8em;color:{color}'>{risk_level}</span>", 
               'font': {'size': 24, 'color': '#FAFAFA'}},
        number={'suffix': "%", 'font': {'size': 60, 'color': '#FAFAFA'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#FAFAFA"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#1F2937",
            'borderwidth': 2,
            'bordercolor': "#374151",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#FAFAFA", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FAFAFA", 'family': "Arial"},
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def create_feature_importance_chart(feature_names, importances):
    """
    Create a horizontal bar chart for feature importance
    
    Args:
        feature_names: List of feature names
        importances: List of importance values
    
    Returns:
        Plotly figure object
    """
    # Sort by importance
    sorted_idx = np.argsort(importances)[-15:]  # Top 15 features
    
    fig = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=importances[sorted_idx],
            colorscale='Viridis',
            line=dict(color='#374151', width=1)
        ),
        text=[f'{imp:.3f}' for imp in importances[sorted_idx]],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={'text': '<b>Feature Importance Analysis</b>', 'font': {'size': 20, 'color': '#FAFAFA'}},
        xaxis_title="Importance Score",
        yaxis_title="Features",
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1F2937',
        font={'color': '#FAFAFA'},
        height=600,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(gridcolor='#374151'),
        yaxis=dict(gridcolor='#374151')
    )
    
    return fig


def create_prediction_distribution(prob_churn):
    """
    Create a bar chart showing prediction distribution
    
    Args:
        prob_churn: Probability of churn
    
    Returns:
        Plotly figure object
    """
    prob_stay = 1 - prob_churn
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Stay', 'Churn'],
            y=[prob_stay * 100, prob_churn * 100],
            marker=dict(
                color=['#10B981', '#EF4444'],
                line=dict(color='#374151', width=2)
            ),
            text=[f'{prob_stay*100:.1f}%', f'{prob_churn*100:.1f}%'],
            textposition='auto',
            textfont=dict(size=18, color='#FAFAFA')
        )
    ])
    
    fig.update_layout(
        title={'text': '<b>Prediction Distribution</b>', 'font': {'size': 20, 'color': '#FAFAFA'}},
        yaxis_title="Probability (%)",
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1F2937',
        font={'color': '#FAFAFA'},
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis=dict(range=[0, 100], gridcolor='#374151'),
        xaxis=dict(gridcolor='#374151')
    )
    
    return fig


def create_roc_curve(fpr_dict, tpr_dict, auc_dict):
    """
    Create ROC curve comparison for multiple models
    
    Args:
        fpr_dict: Dictionary of false positive rates
        tpr_dict: Dictionary of true positive rates
        auc_dict: Dictionary of AUC scores
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = {
        'Logistic Regression': '#3B82F6',
        'Decision Tree': '#10B981',
        'Random Forest': '#F59E0B',
        'XGBoost': '#EF4444'
    }
    
    for model_name in fpr_dict.keys():
        fig.add_trace(go.Scatter(
            x=fpr_dict[model_name],
            y=tpr_dict[model_name],
            mode='lines',
            name=f'{model_name} (AUC = {auc_dict[model_name]:.3f})',
            line=dict(color=colors.get(model_name, '#FFFFFF'), width=3)
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#6B7280', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title={'text': '<b>ROC Curve Comparison</b>', 'font': {'size': 20, 'color': '#FAFAFA'}},
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1F2937',
        font={'color': '#FAFAFA'},
        height=600,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(gridcolor='#374151', range=[0, 1]),
        yaxis=dict(gridcolor='#374151', range=[0, 1]),
        legend=dict(
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.95,
            bgcolor='rgba(31, 41, 55, 0.8)',
            bordercolor='#374151',
            borderwidth=1
        )
    )
    
    return fig


def create_confusion_matrix_heatmap(cm, model_name):
    """
    Create a confusion matrix heatmap
    
    Args:
        cm: Confusion matrix array
        model_name: Name of the model
    
    Returns:
        Plotly figure object
    """
    labels = ['Stay', 'Churn']
    
    # Create annotations
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(cm[i, j]),
                    font=dict(size=24, color='white' if cm[i, j] > cm.max()/2 else 'black'),
                    showarrow=False
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdYlGn_r',
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=24, color='white'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={'text': f'<b>Confusion Matrix - {model_name}</b>', 'font': {'size': 20, 'color': '#FAFAFA'}},
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1F2937',
        font={'color': '#FAFAFA'},
        height=500,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    return fig


def generate_pdf_report(customer_data, prediction, probability, risk_level, top_features):
    """
    Generate a PDF report for customer prediction
    
    Args:
        customer_data: Dictionary of customer information
        prediction: Prediction result (0 or 1)
        probability: Churn probability
        risk_level: Risk level string
        top_features: List of tuples (feature_name, importance_value)
    
    Returns:
        BytesIO object containing PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#374151'),
        spaceAfter=12
    )
    
    # Title
    story.append(Paragraph("Customer Churn Prediction Report", title_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Prediction Result
    story.append(Paragraph("Prediction Summary", heading_style))
    
    result_text = "Will Churn" if prediction == 1 else "Will Stay"
    result_color = colors.red if prediction == 1 else colors.green
    
    result_data = [
        ['Prediction', result_text],
        ['Churn Probability', f'{probability*100:.2f}%'],
        ['Risk Level', risk_level]
    ]
    
    result_table = Table(result_data, colWidths=[2.5*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(result_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # Customer Details
    story.append(Paragraph("Customer Information", heading_style))
    
    customer_table_data = [[k, str(v)] for k, v in customer_data.items()]
    customer_table = Table(customer_table_data, colWidths=[2.5*inch, 3*inch])
    customer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(customer_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # Top Contributing Features
    story.append(Paragraph("Top Contributing Features", heading_style))
    
    feature_data = [['Feature', 'Importance']] + [[f, f'{imp:.4f}'] for f, imp in top_features[:10]]
    feature_table = Table(feature_data, colWidths=[3*inch, 2.5*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F2937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
    ]))
    
    story.append(feature_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def get_risk_level(probability):
    """
    Determine risk level based on churn probability
    
    Args:
        probability: Churn probability (0-1)
    
    Returns:
        Risk level string
    """
    if probability < 0.4:
        return "Low Risk"
    elif probability < 0.75:
        return "Medium Risk"
    else:
        return "High Risk"


def apply_custom_css():
    """
    Apply custom CSS styling to Streamlit app
    """
    custom_css = """
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 100%);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        /* Headers */
        h1 {
            background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        h2, h3 {
            color: #FAFAFA;
            font-weight: 700;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(255, 75, 75, 0.3);
        }
        
        /* File uploader */
        .stFileUploader {
            border: 2px dashed #374151;
            border-radius: 10px;
            padding: 2rem;
            background: #1F2937;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1F2937;
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #1F2937;
            border-radius: 10px;
            font-weight: 600;
        }
        
        /* Success/Error boxes */
        .stSuccess {
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10B981;
        }
        
        .stError {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #EF4444;
        }
        
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #F59E0B;
        }
        
        /* Divider */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #374151, transparent);
            margin: 2rem 0;
        }
    </style>
    """
    return custom_css
