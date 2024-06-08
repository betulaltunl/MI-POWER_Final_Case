from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def remove_index_column(df):
    if 'index' in df.columns:
        df.drop('index', axis=1, inplace=True)
        print("'index' sütunu veri setinden silindi.")
    else:
        print("'index' sütunu veri setinde bulunamadı veya indeks sütunu zaten mevcut değil.")
    return df



def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car, num_but_cat


def get_head_and_tail(df):
    head_html = df.head().to_html(classes='table table-striped')
    tail_html = df.tail().to_html(classes='table table-striped')
    return head_html, tail_html

def create_pie_chart(df, target_column):
    counts = df[target_column].value_counts()
    total = counts.sum()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, 
        values=counts.values,
        hoverinfo='label+percent', 
        textinfo='label+percent',
        texttemplate='%{label}: %{percent:.1f} (%{value:d})'
    )])
    fig.update_layout(title=f"{target_column} Distribution")
    return pio.to_html(fig, full_html=False)


def create_boxplot(df, num_cols):
    fig = px.box(df, y=num_cols, title="Boxplot of Variables")
    fig.update_layout(xaxis_title='Variables', yaxis_title='Values')
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

        # İlk kutu grafiği (aykırı değerler dahil)
        fig = px.box(df, y=col, title=f'{col} Outlier Analysis (IQR Method)')
        fig.add_shape(type="line", x0=0, x1=1, y0=lower_bound, y1=lower_bound, line=dict(color="red", width=2))
        fig.add_shape(type="line", x0=0, x1=1, y0=upper_bound, y1=upper_bound, line=dict(color="red", width=2))
        plots.append(pio.to_html(fig, full_html=False))

        # Aykırı değerleri min-max normalizasyonu ile dönüştür
        df_normalized = df.copy()
        scaler = MinMaxScaler()
        df_normalized[col] = scaler.fit_transform(df_normalized[col].values.reshape(-1, 1))

        # İkinci kutu grafiği (aykırı değerler dönüştürülmüş)
        fig = px.box(df_normalized, y=col, title=f'{col} Outlier Analysis (IQR Method - Normalized)')
        fig.add_shape(type="line", x0=0, x1=1, y0=scaler.transform([[lower_bound]])[0][0], 
                     y1=scaler.transform([[lower_bound]])[0][0], line=dict(color="red", width=2))
        fig.add_shape(type="line", x0=0, x1=1, y0=scaler.transform([[upper_bound]])[0][0], 
                     y1=scaler.transform([[upper_bound]])[0][0], line=dict(color="red", width=2))

        plots.append(pio.to_html(fig, full_html=False))
    return plots

# İNDEX.HTML SAYFASI İÇİN---------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')
#----------------------------------

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

     # 'index' sütununu sil
    df = remove_index_column(df)
    
    # Eğer 'Heart Disease' kolonu yoksa, ilk sütunu hedef olarak kabul edebiliriz.
    if 'Heart Disease' not in df.columns:
        target_column = df.columns[0]
    else:
        target_column = 'Heart Disease'
    
    if df[target_column].dtype == 'O':
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])
    
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    
    num_samples = df.shape[0]
    num_features = df.shape[1]

    pie_chart = create_pie_chart(df, target_column)
    boxplot = create_boxplot(df, num_cols)
    correlation_heatmap = create_correlation_heatmap(df, num_cols)
    cat_summary_plots = create_cat_summary_plots(df, cat_cols)
    num_summary_plots = create_num_summary_plots(df, num_cols)
    target_cat_plots = create_target_cat_plots(df, target_column, cat_cols)
    target_num_plots = create_target_num_plots(df, target_column, num_cols)
    outlier_plots = create_outlier_plots(df, num_cols)
    missing_values_plot = create_missing_values_plot(df)  
    head_html, tail_html = get_head_and_tail(df)

    X = df.drop(target_column, axis=1)  
    y = df[target_column]  
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Eğitim seti üzerinde tahmin yapma
    y_train_pred = model.predict(X_train)

    # Eğitim seti performansı
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    accuracy = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    # Confusion Matrix oluşturma
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

    plt.title('Confusion Matrix')
    plt.ylabel('Actual', fontsize=16)
    plt.xlabel('Predicted',fontsize=16)
    cm_buf = io.BytesIO()
    cm_fig.savefig(cm_buf, format='png')
    cm_buf.seek(0)
    cm_img_base64 = base64.b64encode(cm_buf.getvalue()).decode('utf-8')
    plt.close(cm_fig)

    report = classification_report(y_test, y_pred)
    
    return render_template('result.html', 
                           num_samples=num_samples,
                           num_features=num_features,
                           num_cols=num_cols, 
                           cat_cols=cat_cols,
                           cat_but_car=cat_but_car, 
                           num_but_cat=num_but_cat,
                           pie_chart=pie_chart,
                           boxplot=boxplot,
                           correlation_heatmap=correlation_heatmap,
                           cat_summary_plots=cat_summary_plots,
                           num_summary_plots=num_summary_plots,
                           target_cat_plots=target_cat_plots,
                           target_num_plots=target_num_plots,
                           outlier_plots=outlier_plots,
                           missing_values_plot=missing_values_plot,
                           head_html=head_html,
                           tail_html=tail_html,
                           train_accuracy=train_accuracy,
                           accuracy=accuracy,
                           confusion_matrix=cm_img_base64,
                           classification_report=report)

if __name__ == '__main__':
    app.run(debug=True)
    
                    