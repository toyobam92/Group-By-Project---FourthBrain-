import streamlit as st
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
#from selenium.webdriver.chrome.options import Options
#from selenium import webdriver
import base64
import altair_viewer
import json
from vega import VegaLite

from datetime import datetime
import os
import pandas as pd
import altair as alt
from altair_saver import save as altair_saver
import boto3
import mlflow.sklearn
import pandas as pd
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklift.metrics import uplift_by_percentile
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import base64
import plotly.graph_objs as go
import plotly.figure_factory as ff
import graphviz
from io import BytesIO
from PIL import Image
from sklift.metrics import qini_curve
from sklift.metrics import uplift_auc_score
from fpdf import FPDF

@st.cache_data
def load_data():
    url_dat = 'https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/feature_eng_data.csv'
    data = pd.read_csv(url_dat)
    
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'poutcome', 'month', 'age_0-25', 'age_25-35', 'age_35-50', 'age_50-100',
    'days_0-7', 'days_7-14', 'days_14-21', 'days_21-31']
    num_cols= ['age', 'balance', 'duration', 'pdays', 'previous']
    exclude_cols = ['treatment_tag', 'conversion']

    return data, cat_cols, num_cols, exclude_cols

def grouped_count_plot():
    st.header("Treatment and Control Visualization")
    data, cat_cols, num_cols, exclude_cols = load_data()

    # Group by treatment_tag, conversion, or both (treatment and conversion)
    groupby_options = ['treatment_tag', 'conversion', 'treatment_tag and conversion']
    selected_groupby = st.selectbox("Select groupby option:", groupby_options)

    if selected_groupby == 'treatment_tag':
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('treatment_tag:N', title='Treatment Tag', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count()', title='Count'),
            color=alt.Color('treatment_tag:N', legend=alt.Legend(title='Treatment Tag'))
        ).interactive()
    elif selected_groupby == 'conversion':
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('conversion:N', title='Conversion', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count()', title='Count'),
            color=alt.Color('conversion:N', legend=alt.Legend(title='Conversion'))
        ).interactive()
    else:  # 'treatment_tag and conversion'
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('conversion:N', title='Conversion', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count()', title='Count'),
            color=alt.Color('treatment_tag:N', legend=alt.Legend(title='Treatment_Tag')),
            order=alt.Order('treatment_tag:N', sort='ascending')
        ).properties(
            width=150,
            height=300
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_legend(
            titleFontSize=14
        ).interactive()


        
    return chart 

def create_categorical_plot(df, col, plot_type):
    if plot_type == 'treatment_tag':
        grouped = df.groupby(['treatment_tag', col]).size().reset_index(name='counts')
        grouped['percentage'] = grouped.groupby(['treatment_tag'])['counts'].apply(lambda x: x / x.sum() * 100)
        chart = alt.Chart(grouped).mark_bar().encode(
            x=alt.X(col + ':N', axis=alt.Axis(title=col)),
            y=alt.Y('percentage:Q', axis=alt.Axis(title='Percentage')),
            color=alt.Color('treatment_tag:N', scale=alt.Scale(scheme='tableau20')),
            column=alt.Column('treatment_tag:N', title='Treatment Tag', spacing=10),
            tooltip=[alt.Tooltip('treatment_tag:N'), alt.Tooltip(col + ':N'), alt.Tooltip('counts:Q'),
                     alt.Tooltip('percentage:Q', format='.1f')]
        ).properties(
            title=f'{col}, grouped by treatment tag with percentage',
            width=300
        ).interactive()

    elif plot_type == 'conversion':
        grouped = df.groupby(['conversion', col]).size().reset_index(name='counts')
        grouped['percentage'] = grouped.groupby(['conversion'])['counts'].apply(lambda x: x / x.sum() * 100)
        chart = alt.Chart(grouped).mark_bar().encode(
            x=alt.X(col + ':N', axis=alt.Axis(title=col)),
            y=alt.Y('percentage:Q', axis=alt.Axis(title='Percentage')),
            color=alt.Color('conversion:N', scale=alt.Scale(scheme='set2')),
            column=alt.Column('conversion:N', title='Conversion', spacing=10),
            tooltip=[alt.Tooltip('conversion:N'), alt.Tooltip(col + ':N'), alt.Tooltip('counts:Q'),
                     alt.Tooltip('percentage:Q', format='.1f')]
        ).properties(
            title=f'{col}, grouped by conversion with percentage',
            width=300
        ).interactive()

    elif plot_type == 'treatment_tag and conversion':
        grouped = df.groupby(['treatment_tag', 'conversion', col]).size().reset_index(name='counts')
        grouped['percentage'] = grouped.groupby(['treatment_tag', 'conversion'])['counts'].apply(
            lambda x: x / x.sum() * 100)
        chart = alt.Chart(grouped).mark_bar().encode(
            x=alt.X(col + ':N', axis=alt.Axis(title=col)),
            y=alt.Y('percentage:Q', axis=alt.Axis(title='Percentage')),
            color=alt.Color('conversion:N', scale=alt.Scale(scheme='set2')),
            column=alt.Column('treatment_tag:N', title='Treatment Tag', spacing=10),
            tooltip=[alt.Tooltip('treatment_tag:N'), alt.Tooltip('conversion:N'), alt.Tooltip(col + ':N'),
                     alt.Tooltip('counts:Q'), alt.Tooltip('percentage:Q', format='.1f')]
        ).properties(
            title=f'{col}, grouped by treatment tag and conversion with percentage',
            width=300
        ).interactive()

    else:
        raise ValueError(f'Invalid plot type: {plot_type}')

    return chart



def create_numerical_plot(df, col, plot_type):
    if plot_type == 'treatment_tag':
        means = df.groupby(['treatment_tag'])[col].mean().reset_index()
        x = 'treatment_tag'
        hue = None
    elif plot_type == 'conversion':
        means = df.groupby(['conversion'])[col].mean().reset_index()
        x = 'conversion'
        hue = None
    elif plot_type == 'treatment_tag and conversion':
        means = df.groupby(['treatment_tag', 'conversion'])[col].mean().reset_index()
        conversion_mapping = {0: 'No Conversion', 1: 'Conversion'}
        treatment_tag_mapping = {0: 'Control', 1: 'Treatment'}
        means['conversion_label'] = means['conversion'].map(conversion_mapping)
        means['treatment_tag_label'] = means['treatment_tag'].map(treatment_tag_mapping)
        means['group'] = means['treatment_tag_label'] + ', ' + means['conversion_label']
        x = 'group'
        hue = None
    else:
        raise ValueError(f'Invalid plot type: {plot_type}')

    chart = alt.Chart(means).mark_bar().encode(
        x=alt.X(x + ':N', axis=alt.Axis(labelAngle=0,labelFontSize=9,title=x)),
        y=alt.Y(col + ':Q', axis=alt.Axis(title='Mean')),
        color=alt.Color(hue + ':N', scale=alt.Scale(scheme='tableau20')) if hue else x + ':N',
        tooltip=[alt.Tooltip(x + ':N')] + [alt.Tooltip(hue + ':N')] if hue else [] + [alt.Tooltip(col + ':Q', format='.2f')]
    ).properties(
        title=f'Mean of {col}, grouped by {plot_type}',
        width=150
    ).interactive()
    

    return chart


def categorical_analysis():
    st.write('Categorical Features')

    data, cat_cols, num_cols, exclude_cols = load_data()
    selected_col = st.selectbox('Select a column', cat_cols)

    st.write('Select View to Group Data')
    plot_type = st.selectbox('', ['treatment_tag', 'conversion', 'treatment_tag and conversion'])

    if st.button('Generate Plot'):
        if plot_type:
            chart = create_categorical_plot(data, selected_col, plot_type)
            st.altair_chart(chart)

def numerical_analysis():
    st.write('Numerical Features')
    data, cat_cols, num_cols, exclude_cols = load_data()

    selected_col = st.selectbox('Select a column', num_cols)
    
    st.write('Select View to Group Data')
    plot_type = st.selectbox('', ['treatment_tag', 'conversion', 'treatment_tag and conversion'])

    if st.button('Generate Plot'):
        if plot_type:
            chart = create_numerical_plot(data, selected_col, plot_type)
            st.altair_chart(chart, use_container_width=True)
  
  
def prepare_data_for_plots(uplift_ct, trmnt_test, y_test, X_test_2):
    # Convert the arrays to pandas Series
    uplift_sm = pd.Series(uplift_ct)
    trmnt_test = pd.Series(trmnt_test['treatment_tag'])
    y_test = pd.Series(y_test['conversion'])

    # Create a new column in the test set dataframe and assign uplift scores to it
    test_set_df = pd.concat([X_test_2.reset_index(drop=True), trmnt_test.reset_index(drop=True), y_test.reset_index(drop=True),  uplift_sm.reset_index(drop=True)], axis=1)
    test_set_df.columns = list(X_test_2.columns) + ['treatment_tag', 'conversion', 'uplift_score']
    
    return test_set_df          
            
@st.cache_data
def campaign_results():

    # Load the model
    loaded_model = get_model_uri()
    
    # Load data
    X_test_2, y_test, trmnt_test = load_data_model()

    # Make predictions
    uplift_ct = loaded_model.predict(X_test_2)

    # Calculate uplift by percentile
    ct_percentile = uplift_by_percentile(y_test, uplift_ct, trmnt_test,
                                         strategy='overall', total=True, std=True, bins=10)
    df = pd.DataFrame(ct_percentile)

    
    plot_data_df = prepare_data_for_plots(uplift_ct, trmnt_test, y_test, X_test_2)

    return df, plot_data_df ,X_test_2, y_test, trmnt_test 

@st.cache_resource
def get_model_uri():
    s3 = boto3.client('s3')
    bucket_name = 'uplift-model'
    model_uri = f's3://{bucket_name}/final/Group-By-Project---FourthBrain-/nb/mlruns/517342746135544475/af4b942074eb430c97be548979749e6b/artifacts/class_transformation_model'
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return  loaded_model

@st.cache_data
def load_data_model():
    base_url = 'https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/'
    X_test_2 = pd.read_csv(f'{base_url}X_test.csv')
    y_test = pd.read_csv(f'{base_url}y_test.csv')
    trmnt_test = pd.read_csv(f'{base_url}trmnt_test.csv')
    return X_test_2, y_test, trmnt_test

def create_bar_chart(df):
    chart = alt.Chart(df.reset_index()).mark_bar().encode(
        x=alt.X('percentile', title='Percentile'),
        y=alt.Y('value:Q', title='Number of participants', scale=alt.Scale(domain=[0, max(df[['n_treatment', 'n_control']].values.flatten())]),stack=True),
        color=alt.Color('variable:N', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title='Group'))
    ).transform_fold(
        ['n_treatment', 'n_control'], 
        as_=['variable', 'value']
    ).transform_calculate(
        variable=alt.expr.if_(alt.datum.variable == 'n_treatment', 'Treatment', 'Control')
    ).properties(
        title='Number of Participants', width=1000, height=400
    ).configure_scale(bandPaddingInner=0,
                  bandPaddingOuter=0.1,
    ).configure_header(labelOrient='bottom',
                   labelPadding = 3).configure_facet(spacing=5
    ).configure_axis(labelAngle=0).interactive()
    
    return chart


def create_line_chart(df):
    chart = alt.Chart(df.reset_index()).mark_line().encode(
        x=alt.X('percentile', title='Percentile', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('value:Q', title='Response rate'),
        color=alt.Color('variable:N', scale=alt.Scale(domain=['response_rate_treatment', 'response_rate_control'], range=['#4C72B0', '#55A868']), legend=alt.Legend(title='Group'))
    ).transform_fold(['response_rate_treatment', 'response_rate_control'], as_=['variable', 'value']).interactive()

    return chart



def create_uplift_chart(df):
    chart = alt.Chart(df.reset_index()).mark_bar().encode(
        x=alt.X('percentile', title='Percentile', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('uplift:Q', title='Uplift'),
        color=alt.condition(
            alt.datum.uplift > 0,
            alt.value("#1f77b4"),  # Positive uplift bars
            alt.value("#d62728")    # Negative uplift bars
        )
    ).properties(title='Uplift by percentile').interactive()
    return chart


def create_scatter_plot_with_regression(df):
    # Scatter plot
    scatter_plot = alt.Chart(df).mark_circle().encode(
        x=alt.X('response_rate_control:Q', title='Control Response Rate'),
        y=alt.Y('response_rate_treatment:Q', title='Treatment Response Rate'),
        color=alt.value('#4C72B0')
    ).properties(width=800, height=400, title='Scatter Plot of Response Rates').interactive()

    # Regression line
    x = df['response_rate_control']
    y = df['response_rate_treatment']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.linspace(0, max(x))
    line_y = slope * line_x + intercept
    regression_df = pd.DataFrame({'x': line_x, 'y': line_y})
    
    regression_line = alt.Chart(regression_df).mark_line(color='black', strokeDash=[3,3]).encode(
        x=alt.X('x:Q'),
        y=alt.Y('y:Q')
    )

    return scatter_plot + regression_line

def create_box_plot(df):
    # Melt the DataFrame to have a long format
    df_melted = df[['response_rate_control', 'response_rate_treatment']].melt(var_name='Group', value_name='Response Rate')

    # Create the box plot
    box_plot = alt.Chart(df_melted).mark_boxplot(size=100).encode(
        x=alt.X('Group:N', title=None, axis=alt.Axis(labelAngle=0,labelFontSize=8)),
        y=alt.Y('Response Rate:Q', title='Response Rate'),
        color=alt.Color('Group:N', scale=alt.Scale(range=['#55A868', '#4C72B0']))
    ).properties(width=400, height=400, title='Box Plot of Response Rates').configure_axisY(labelPadding=15).interactive()
    
    return box_plot



def clean():
    
    df, plot_data_df ,X_test_2, y_test, trmnt_test  = campaign_results()
    
    initial_quartile_values = [0, 0.2, 0.5, 0.8, 1]
    quartile_values = [st.sidebar.number_input(f'Value for {i}% quartile', min_value=0.0, max_value=1.0, value=float(initial_quartile_values[i]), step=0.1) for i in range(5)]
    
    label_names = ['Sleeping Dogs', 'Lost Causes', 'Sure Things', 'Persuadables']
    plot_data_df['uplift_category'] = pd.qcut(plot_data_df['uplift_score'], q=quartile_values, labels=label_names, duplicates='drop')
    
    qini_x, qini_y = qini_curve(y_test, plot_data_df['uplift_score'], trmnt_test)
    
    return plot_data_df, (qini_x, qini_y), y_test, trmnt_test


def uplift_histogram(df):
    hist = alt.Chart(df).mark_bar().encode(
        alt.X('uplift_score:Q', bin=alt.Bin(step=0.03)),
        alt.Y('count():Q'),
        alt.Color('uplift_category:N', scale=alt.Scale(scheme='category10')),
        tooltip=['uplift_category', 'count()']
    ).properties(
        width=500,
        height=300,
        title='Uplift Score Distribution by Uplift Category'
    ).interactive()
    return hist

def uplift_count_plot(df):
    count_plot = alt.Chart(df).mark_bar().encode(
        x=alt.X('uplift_category:N', axis=alt.Axis(labelAngle=0)),
        y='count():Q',
        color=alt.Color('uplift_category:N', scale=alt.Scale(scheme='category10')),
        tooltip=['uplift_category', 'count()']
    ).properties(
        width=500,
        height=300,
        title='Uplift Categories'
    ).interactive()
    return count_plot


def uplift_bar_plot(df):
    uplift_scores = df.groupby('uplift_category').mean()[['uplift_score']]
    uplift_bar = alt.Chart(uplift_scores.reset_index()).mark_bar().encode(
        x=alt.X('uplift_category:N', axis=alt.Axis(labelAngle=0)),
        y='uplift_score:Q',
        color=alt.Color('uplift_category:N', scale=alt.Scale(scheme='category10')),
        tooltip=['uplift_category', 'uplift_score']
    ).properties(
        width=500,
        height=300,
        title='Average Uplift Score by Uplift Category'
    ).interactive()
    return uplift_bar

def decision_tree_plot(df):
    # Create decision tree classifier and fit to data
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(df.drop(['uplift_category', 'uplift_score'], axis=1), df['uplift_category'])

    # Plot decision tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, filled=True, feature_names=df.drop(['uplift_category', 'uplift_score'], axis=1).columns, class_names=tree.classes_)
    plt.title("Decision Tree")

    return fig


def create_uplift_cat_countplot(df):
    variable = st.selectbox('Select variable', ['job_admin.', 'job_blue-collar',
        'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
        'job_self-employed', 'job_services', 'job_student', 'job_technician',
        'job_unemployed', 'marital_divorced', 'marital_married',
        'marital_single', 'education_primary', 'education_secondary',
        'education_tertiary', 'default_no', 'default_yes', 'housing_no',
        'housing_yes', 'loan_no', 'loan_yes', 'contact_cellular',
        'contact_telephone', 'poutcome_failure', 'poutcome_success',
        'month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
        'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
        'month_oct', 'month_sep', 'age_0-25', 'age_25-35', 'age_35-50',
        'age_50-100', 'days_0-7', 'days_7-14', 'days_14-21', 'days_21-31',
        'treatment_tag', 'conversion'])
    grouped = df.groupby('uplift_category')[variable].value_counts().unstack(fill_value=0)
    grouped = grouped.reset_index().melt(id_vars='uplift_category', var_name='value', value_name='count')
    # Create the Altair plot
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X('uplift_category:N', axis=alt.Axis(title='Uplift category')),
        y=alt.Y('count:Q', axis=alt.Axis(title=f'{variable} count')),
        color=alt.Color('value:N', legend=alt.Legend(title='Value')),
        order=alt.Order('value:N')
    ).properties(
        width=800,
        height=400,
        title=f'{variable} counts by uplift category'
    ).interactive()
    chart = chart.configure_axis(labelAngle=0)
    return chart

def explore_predicted_observations(df):
    category = st.selectbox('Select category', df['uplift_category'].unique())
    category_df = df[df['uplift_category'] == category]
    csv = category_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{category.lower()}.csv">Download {category} Data</a>'
    return category,href, category_df


class CustomPDF(FPDF):
    def titles(self, title, subtitle, date):
        self.set_font("Arial", "B", size=16)
        self.cell(190, 10, txt=title, ln=True, align="C")
        self.set_font("Arial", "", size=14)
        self.cell(190, 10, txt=subtitle, ln=True, align="C")
        self.cell(190, 10, txt=date, ln=True, align="C")
        self.ln(20)

    def chapter_title(self, title):
        self.set_font("Arial", "B", size=14)
        self.cell(190, 10, txt=title, ln=True, align="L")
        self.ln(5)

    def plot_image(self, image_path):
        self.image(image_path, x=10, w=180)
        self.ln(10)

    def description(self, text):
        self.set_font("Arial", "", size=12)
        self.multi_cell(190, 10, txt=text, align="L")
        self.ln(5)

def save_plots_and_generate_report(plot_data_df):
    plots = {
        'Uplift Histogram': {
            'filename': 'uplift_histogram.png',
            'func': uplift_histogram,
        },
        'Uplift Count Plot': {
            'filename': 'uplift_count_plot.png',
            'func': uplift_count_plot,
        },
        'Uplift Bar Plot': {
            'filename': 'uplift_bar_plot.png',
            'func': uplift_bar_plot,
        },
        'Decision Tree Plot': {
            'filename': 'decision_tree_plot.png',
            'func': decision_tree_plot,
        },
    }

    # Save the plots as images
    for plot_title, plot_info in plots.items():
        chart = plot_info['func'](plot_data_df)
        
        if plot_title == 'Decision Tree Plot':
            plt.savefig(plot_info['filename'])
            plt.close()
        else:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("window-size=1400,1500")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("start-maximized")
            options.add_argument("enable-automation")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(options=options)
            chart.save(plot_info['filename'], driver = driver)

    # Create the PDF report
    pdf = CustomPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add cover page
    today = datetime.now().strftime("%Y-%m-%d")
    pdf.titles("Uplift Model Report", "Subtitle: Additional Information", f"Date: {today}")

    # Add table of contents
    pdf.chapter_title("Table of Contents")
    for i, plot_title in enumerate(plots):
        pdf.cell(0, 5, f"{i + 1}. {plot_title}", ln=True)

    # Add plots to the PDF report
    for i, (plot_title, plot_info) in enumerate(plots.items()):
        pdf.chapter_title(f"{i + 1}. {plot_title}")
        pdf.image(plot_info['filename'], w=pdf.get_page_width() - 40, h=0)

    # Save the PDF report
    report_filename = "uplift_model_report.pdf"
    pdf.output(report_filename)

    # Remove the image files
    for plot_info in plots.values():
        os.remove(plot_info['filename'])
        
    b64_pdf = base64.b64encode(report_filename.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="report.pdf">Download PDF</a>'
    return href

def save_plots_and_generate_report(plot_data_df, driver):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=1400,1500")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("enable-automation")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    plots = {
        'Uplift Histogram': {'func': uplift_histogram, 'filename': 'uplift_histogram.png'},
        'Uplift Count Plot': {'func': uplift_count_plot, 'filename': 'uplift_count_plot.png'},
        'Uplift Bar Plot': {'func': uplift_bar_plot, 'filename': 'uplift_bar_plot.png'},
        'Decision Tree Plot': {'func': decision_tree_plot, 'filename': 'decision_tree_plot.png'}
    }

    # Save the plots as images
    for plot_title, plot_info in plots.items():
        chart_function = plot_info['func']
        chart = chart_function(plot_data_df)
        
        if plot_title == 'Decision Tree Plot':
            plt.savefig(plot_info['filename'])
            plt.close()
        else:
            chart.save(plot_info['filename'], driver=driver)

    # Create the PDF report
    pdf = CustomPDF(driver)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add cover page
    today = datetime.now().strftime("%Y-%m-%d")
    pdf.titles("Uplift Model Report", "Subtitle: Additional Information", f"Date: {today}")

    # Add table of contents
    pdf.chapter_title("Table of Contents")
    for i, plot_title in enumerate(plots):
        pdf.cell(0, 5, f"{i + 1}. {plot_title}", ln=True)

    # Add plots to the PDF report
    for i, (plot_title, plot_info) in enumerate(plots.items()):
        pdf.chapter_title(f"{i + 1}. {plot_title}")
        pdf.image(plot_info['filename'], w=pdf.get_page_width() - 40, h=0)

    # Save the PDF report
    report_filename = "uplift_model_report.pdf"
    pdf.output(report_filename)

    # Remove the image files
    for plot_info in plots.values():
        os.remove(plot_info['filename'])

    return report_filename


def plot_qini_curve(qini_x, qini_y ,y_test, trmnt_test, plot_data_df):
    
    auc_val = uplift_auc_score(y_test, pd.Series( plot_data_df['uplift_score'] ), trmnt_test)
    
    qini_data = pd.DataFrame({'Percentage of data targeted': qini_x, 'Qini': qini_y})

    # Calculate random and perfect lines
    qini_data['Random'] = qini_data['Percentage of data targeted'] * qini_y[-1] / qini_x[-1]
    qini_data['Perfect'] = qini_data['Qini'].cummax()

    # Create a DataFrame for each line with an additional 'Line' column
    qini_curve_data = qini_data.melt(id_vars=['Percentage of data targeted'], value_vars=['Qini', 'Random', 'Perfect'],
                                      var_name='Line', value_name='Uplift')

    # Create the line chart with a legend
    chart = alt.Chart(qini_curve_data).mark_line().encode(
        x='Percentage of data targeted',
        y='Uplift',
        color=alt.Color('Line', legend=alt.Legend(title='Lines')),
        strokeDash=alt.condition(alt.datum.Line == 'Qini', alt.value([1, 0]), alt.value([3, 3]))
    ).properties(
        title=f'Qini Curve (AUC: {auc_val})'
    ).interactive()
#:.4f
    return chart

def welcome_page():
    st.title("Welcome to the Uplift Model Platform")
    st.write("""
    The Uplift Model Platform is designed to help businesses and individuals evaluate the effectiveness of their marketing campaigns, promotions, and customer engagement strategies. With our cutting-edge uplift modeling techniques, you can optimize your marketing efforts by identifying the most suitable target audience and maximizing return on investment.
    
    In this platform, you'll be able to:
    * Explore your campaign data
    * Evaluate the Uplift Model's Performance
    * Visualize and analyze the Model and Segmentation results
    * Identify your Persudables, Sure Things, Lost Causes and Sleeping dogs
    * Make data-driven decisions to improve your marketing strategy
    
    Let's get started! Choose an option from the radio button menu on the left to navigate through the platform.
    """)  
    
def main():
    st.set_page_config(page_title='Uplift Model', page_icon=':bar_chart:')
    st.title('Uplift Model - Campaign Analytics')
    
    tabs = ['Welcome', 'Campaign Visualizations','Exploratory Data Analysis', 'Campaign Results', 'Uplift Segment']
    selected_tab = st.sidebar.radio('', tabs)

    if selected_tab == 'Exploratory Data Analysis':
        st.write('Select analysis type:')
        analysis_type = st.selectbox('', ['Categorical Analysis', 'Numerical Analysis'])
        if analysis_type == 'Categorical Analysis':
            categorical_analysis()
        elif analysis_type == 'Numerical Analysis':
            numerical_analysis()
            
 
    elif selected_tab == 'Campaign Results':
        df, plot_data_df ,X_test_2, y_test, trmnt_test = campaign_results()
             # Create a selection box for the plots
        plot_options = [
        'Uplift Chart',
        'Scatter Plot',
        'Box Plot',
        'Line Chart',
        'Bar Chart' ]
        selected_plot = st.selectbox('Select a plot to display:', plot_options) 
        # Create and display the selected plot
        if selected_plot == 'Uplift Chart':
            uplift_chart = create_uplift_chart(df)
            st.altair_chart(uplift_chart, use_container_width=True)
        elif selected_plot == 'Scatter Plot':
            scatter_plot = create_scatter_plot_with_regression(df )
            st.altair_chart(scatter_plot, use_container_width=True)
        elif selected_plot == 'Box Plot':
            box_plot = create_box_plot(df )
            st.altair_chart(box_plot, use_container_width=True)
        elif selected_plot == 'Line Chart':
            line_chart = create_line_chart(df)
            st.altair_chart(line_chart, use_container_width=True)
        elif selected_plot == 'Bar Chart':
            bar_chart = create_bar_chart(df)
            st.altair_chart(bar_chart, use_container_width=True)
    elif selected_tab == 'Uplift Segment':
        plot_data_df, (qini_x, qini_y), y_test, trmnt_test = clean()
        plot_options = [
        'Uplift Histogram',
        'Uplift Count Plot',
        'Uplift Bar Plot',
        'Decision Tree Plot',
        'Qini Curve',
        'Uplift by Variable',
        'Explore and Download Predicted Observations']
        selected_plot = st.selectbox('Select a plot to display:', plot_options) 
        if selected_plot == 'Uplift Histogram':
            plot = uplift_histogram(plot_data_df)
            st.altair_chart(plot, use_container_width=True)
        elif selected_plot == 'Uplift Count Plot': 
            plot = uplift_count_plot(plot_data_df)
            st.altair_chart(plot, use_container_width=True)
        elif selected_plot == 'Uplift Bar Plot': 
            plot = uplift_bar_plot(plot_data_df)
            st.altair_chart(plot, use_container_width=True)
        elif selected_plot == 'Decision Tree Plot': 
            st.pyplot(decision_tree_plot(plot_data_df))
        elif selected_plot == 'Uplift by Variable': 
            plot = create_uplift_cat_countplot(plot_data_df)  
            st.altair_chart(plot, use_container_width=True)
        elif selected_plot == 'Explore and Download Predicted Observations': 
            category,href,category_df = explore_predicted_observations(plot_data_df)
            st.write(f'Explore Predicted Observations for {category} category')
            st.write(category_df)
            st.markdown(href, unsafe_allow_html=True)
        elif selected_plot == 'Qini Curve':   
           plot =  plot_qini_curve(qini_x, qini_y, y_test, trmnt_test, plot_data_df)
           st.altair_chart(plot, use_container_width=True)
        #elif selected_plot == 'Generate Report':
          # href = save_plots_and_generate_report(plot_data_df)
          # st.markdown(href, unsafe_allow_html=True)
    elif selected_tab == 'Welcome':
        welcome_page()
    elif selected_tab == "Campaign Visualizations":
        plot = grouped_count_plot()
        st.altair_chart(plot, use_container_width=True)
        

        
if __name__ == '__main__':
    main()


