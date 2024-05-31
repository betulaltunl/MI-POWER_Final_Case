from flask import Flask, render_template, request, redirect, url_for

import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder
import os


    

    

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_head_and_tail(df):
    head_html = df.head().to_html(classes='table table-striped')
    tail_html = df.tail().to_html(classes='table table-striped')
    return head_html, tail_html


def create_pie_chart(df):
    counts = df["Heart Disease"].value_counts()
    total = counts.sum()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, 
        values=counts.values,
        hoverinfo='label+percent', 
        textinfo='label+percent',
        texttemplate='%{label}: %{percent:.1f} (%{value:d})'
    )])
    fig.update_layout(title="Heart Disease Distribution")
    return pio.to_html(fig, full_html=False)

def create_boxplot(df):
    fig = px.box(df, title="Boxplot of Variables")
    fig.update_layout(xaxis_title='Variables', yaxis_title='Values')
    return pio.to_html(fig, full_html=False)

def create_unique_counts_plot(df):
    unique_counts = df.nunique()
    fig = px.bar(unique_counts, title='Her Sütundaki Benzersiz Değer Sayıları')
    fig.update_layout(xaxis_title='Sütunlar', yaxis_title='Benzersiz Değer Sayısı')
    return pio.to_html(fig, full_html=False)

def create_missing_values_plot(df):
    missing_values_count = df.isnull().sum()
    missing_values_count = missing_values_count[missing_values_count > 0]  
    if not missing_values_count.empty:  
        fig = px.bar(missing_values_count, x=missing_values_count.index, y=missing_values_count.values,
                     title='Missing Values Count', labels={'x': 'Columns', 'y': 'Missing Values Count'})
        return pio.to_html(fig, full_html=False)
    return '<p>No missing values found.</p>'  


def create_correlation_heatmap(df, num_cols):
    num_df = df[num_cols]  
    corr_matrix = num_df.corr()  
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.index.values,
        y=corr_matrix.columns.values,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
    ))
    fig.update_layout(title='Correlation Heatmap', xaxis=dict(title='Features'), yaxis=dict(title='Features'))
    return pio.to_html(fig, full_html=False)


def create_cat_summary_plots(df, cat_cols):
    plots = []
    for col in cat_cols:
        summary_df = df[col].value_counts().reset_index()
        summary_df.columns = ['value', 'count']
        summary_df['percentage'] = 100 * summary_df['count'] / len(df)
        summary_df['percentage'] = summary_df['percentage'].round(2)
        fig = px.bar(summary_df, x='value', y='count', text='percentage',
                     title=f'{col} Count and Percentage', labels={'value': col, 'count': 'Count', 'percentage': 'Percentage'})
        plots.append(pio.to_html(fig, full_html=False))
    return plots

def create_num_summary_plots(df, num_cols):
    plots = []
    for col in num_cols:
        fig_hist = px.histogram(df, x=col, nbins=30, title=f'{col} Distribution')
        fig_box = px.box(df, y=col, title=f'{col} Box Plot')
        plots.append(pio.to_html(fig_hist, full_html=False))
        plots.append(pio.to_html(fig_box, full_html=False))
    return plots


def create_target_cat_plots(df, target, cat_cols):
    cat_cols.remove(target)  
    plots = []
    for col in cat_cols:
        summary_df = df.groupby(col)[target].mean().reset_index()
        fig = px.bar(summary_df, x=col, y=target, title=f'{col} vs. {target}', labels={col: col, target: target})
        plots.append(pio.to_html(fig, full_html=False))
    return plots


def create_target_num_plots(df, target, num_cols):
    plots = []
    for col in num_cols:
        summary_df = df.groupby(target)[col].mean().reset_index()
        fig = px.bar(summary_df, x=target, y=col, title=f'{target} vs. Mean of {col}', labels={target: target, col: 'Mean of ' + col})
        plots.append(pio.to_html(fig, full_html=False))
    return plots

def create_outlier_plots(df, num_cols):
    plots = []
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        fig = px.box(df, y=col, title=f'{col} Aykırı Değer Analizi (IQR Method)')
        fig.add_shape(type="line", x0=0, x1=1, y0=lower_bound, y1=lower_bound, line=dict(color="red", width=2))
        fig.add_shape(type="line", x0=0, x1=1, y0=upper_bound, y1=upper_bound, line=dict(color="red", width=2))
        plots.append(pio.to_html(fig, full_html=False))
    return plots

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car, num_but_cat







@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    df.drop(['index'], axis=1, inplace=True, errors='ignore')
    label_encoder = LabelEncoder()
    df['Heart Disease'] = label_encoder.fit_transform(df['Heart Disease'])
    
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    
    num_samples = df.shape[0]
    num_features = df.shape[1]
    

   
    matplotlib.use('Agg')

    pie_chart = create_pie_chart(df)
    boxplot = create_boxplot(df)
    unique_counts_plot = create_unique_counts_plot(df)
    correlation_heatmap = create_correlation_heatmap(df, num_cols)
    cat_summary_plots = create_cat_summary_plots(df, cat_cols)
    num_summary_plots = create_num_summary_plots(df, num_cols)
    target_cat_plots = create_target_cat_plots(df, 'Heart Disease', cat_cols)
    target_num_plots = create_target_num_plots(df, 'Heart Disease', num_cols)
    outlier_plots = create_outlier_plots(df, num_cols)
    missing_values_plot = create_missing_values_plot(df)  
    head_html, tail_html = get_head_and_tail(df)


 
    X = df.drop('Heart Disease', axis=1)  
    y = df['Heart Disease']  
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    
    accuracy = model.score(X_test, y_test)
    
   
    y_pred = model.predict(X_test)
    
    
    cm = confusion_matrix(y_test, y_pred)
    
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    plt.close()
    
    return render_template('result.html', num_samples=num_samples, num_features=num_features, cat_cols=cat_cols, 
                           num_cols=num_cols, cat_but_car=cat_but_car, num_but_cat=num_but_cat, pie_chart=pie_chart, 
                           boxplot=boxplot, unique_counts_plot=unique_counts_plot, correlation_heatmap=correlation_heatmap, 
                           cat_summary_plots=cat_summary_plots, num_summary_plots=num_summary_plots, 
                           target_cat_plots=target_cat_plots, target_num_plots=target_num_plots, outlier_plots=outlier_plots,
                           missing_values_plot=missing_values_plot, head_html=head_html, tail_html=tail_html, accuracy=accuracy, graphic=graphic)

if __name__ == "__main__":
    app.run(debug=True)
