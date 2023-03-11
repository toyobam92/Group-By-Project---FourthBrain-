import streamlit as st
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import os
from sklift.metrics import uplift_by_percentile
from scipy.stats import linregress
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree


categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'month']
other_bins = ['age_0-25', 'age_25-35', 'age_35-50', 'age_50-100', 'days_0-7', 'days_7-14', 'days_14-21', 'days_21-31']
categorical_cols = categorical_features + other_bins
numerical_cols = ['age', 'balance', 'duration', 'pdays', 'previous']

plt.rcParams['font.family'] = 'serif'
plt.style.use('seaborn-pastel')


def load_data():
    url_dat = 'https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/feature_eng_data.csv'
    return pd.read_csv(url_dat)

def create_plot(df, col, plot_type):
    if plot_type == 'treatment_tag':
        grouped = df.groupby(['treatment_tag', col]).size().reset_index(name='counts')
        grouped['percentage'] = grouped.groupby(['treatment_tag'] )['counts'].apply(lambda x: x / x.sum() * 100)
        grouped = grouped.pivot(index=col, columns=['treatment_tag'], values=['counts', 'percentage'])
        ax = grouped['percentage'].plot.bar(stacked=False, legend=True)

        ax.set_title(f'{col}, treatment with percentage')
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage', labelpad=15)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge', labels=[f'{int(round(val))}%' for val in p.datavalues], fontsize=8)

        ax.legend(fontsize=8)
        labels = [ 'Control', 'Treatment']
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8)

        return ax

    elif plot_type == 'conversion':
        grouped = df.groupby(['conversion', col]).size().reset_index(name='counts')
        grouped['percentage'] = grouped.groupby(['conversion'] )['counts'].apply(lambda x: x / x.sum() * 100)
        grouped = grouped.pivot(index=col, columns=['conversion'], values=['counts', 'percentage'])
        ax = grouped['percentage'].plot.bar(stacked=False, legend=True)

        ax.set_title(f'{col}, conversion with percentage')
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage', labelpad=15)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge', labels=[f'{int(round(val))}%' for val in p.datavalues], fontsize=8)

        ax.legend(fontsize=8)
        labels = ['Conversion', 'No Conversion']
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8)

        return ax

    elif plot_type == 'treatment_tag and conversion':
        grouped = df.groupby(['treatment_tag', 'conversion', col]).size().reset_index(name='counts')
        grouped['percentage'] = grouped.groupby(['treatment_tag','conversion'] )['counts'].apply(lambda x: x / x.sum() * 100)
        grouped = grouped.pivot(index=col, columns=['treatment_tag','conversion'], values=['counts', 'percentage'])
        ax = grouped['percentage'].plot.bar(stacked=False, legend=True)

        ax.set_title(f'{col}, treatment, and conversion with percentage')
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage', labelpad=15)

        for p in ax.containers:
            ax.bar_label(p, label_type='edge', labels=[f'{int(round(val))}%' for val in p.datavalues], fontsize=8)

        ax.legend(fontsize=8)
        labels = ['No Conversion - Control', 'Conversion - Control', 'No Conversion - Treatment', 'Conversion - Treatment']
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8)
    

        return ax
    

    elif plot_type == 'treatment_tag and conversion numerical':
        group_cols = ['treatment_tag', 'conversion']
        grouped = df.groupby(['treatment_tag','conversion'] )[col].mean().round(2).reset_index()
        ax = sns.barplot(x='treatment_tag', y=col, hue='conversion', data=grouped)

        return ax

    else:
        raise ValueError(f'Invalid plot type: {plot_type}')

def categorical_analysis():
    st.write('Categorical Features')

    data = load_data()
    selected_col = st.selectbox('Select a column', categorical_cols)
    
    st.write('Select View to Group Data')
    plot_type = st.selectbox('', ['treatment_tag', 'conversion', 'treatment_tag and conversion'])

    if st.button('Generate Plot'):
        if plot_type:
            ax = create_plot(data, selected_col, plot_type)
            st.pyplot(ax.figure)

def numerical_analysis():
    st.write('Numerical Features')
    data = load_data()
    
    selected_col = st.selectbox('Select a column', numerical_cols)
    
    st.write('Select View to Group Data')
    plot_type = st.selectbox('', ['treatment_tag and conversion numerical'])

    if st.button('Generate Plot'):
        if plot_type:
            ax = create_plot(data, selected_col, plot_type)
            st.pyplot(ax.figure)

def campaign_results():
        #st.write(os.listdir())
        st.write('Campaign Results')
        
        s3 = boto3.client('s3')
        bucket_name = 'uplift-model'
        model_uri = 's3://{}/final/Group-By-Project---FourthBrain-/nb/mlruns/517342746135544475/af4b942074eb430c97be548979749e6b/artifacts/class_transformation_model'.format(bucket_name)

# Load the model
        loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)
        # Replace with the actual path to the MLflow model
        #model_uri = "nb/mlruns/517342746135544475/af4b942074eb430c97be548979749e6b/artifacts/class_transformation_model"
        
        # Load the model from the run
        #loaded_model = mlflow.sklearn.load_model(model_uri)
        
        X_test_2 = pd.read_csv('https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/X_test.csv')
        y_test = pd.read_csv('https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/y_test.csv')
        trmnt_test = pd.read_csv('https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/trmnt_test.csv')

        uplift_ct = loaded_model.predict(X_test_2)

        from sklift.metrics import uplift_by_percentile
        ct_percentile = uplift_by_percentile(y_test, uplift_ct, trmnt_test,
                                            strategy='overall',
                                            total=True, std=True, bins=10)

        df = pd.DataFrame(ct_percentile)

        # Set the default style
        sns.set_style('whitegrid')

        # Bar chart of the number of participants
        fig, axs = plt.subplots(figsize=(20, 10))
        df[['n_treatment', 'n_control']].plot(kind='bar', stacked=False, ax=axs, width=0.8, color=['#4C72B0', '#55A868'])
        axs.set_xlabel('Percentile', fontsize=14)
        axs.set_ylabel('Number of participants', fontsize=14)
        axs.tick_params(axis='both', labelsize=10)
        axs.legend(fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        # Line chart of the response rates
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].plot(df.index, df['response_rate_treatment'], label='Treatment', color='#4C72B0', linewidth=2)
        axs[0].plot(df.index, df['response_rate_control'], label='Control', color='#55A868', linewidth=2)
        axs[0].set_xlabel('Percentile', fontsize=14)
        axs[0].set_ylabel('Response rate', fontsize=14)
        axs[0].tick_params(axis='both', labelsize=10)
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].legend(fontsize=12)

        # Bar chart of the uplift
        axs[1].bar(df.index, df['uplift'], color='#4C72B0')
        axs[1].set_xlabel('Percentile', fontsize=14)
        axs[1].set_ylabel('Uplift', fontsize=14)
        axs[1].tick_params(axis='both', labelsize=10)
        axs[1].tick_params(axis='x', rotation=45)
        # Adjust the position of the y-axis label
        axs[1].yaxis.set_label_coords(-0.15, 0.5)

        # Scatter plot of the response rates
        axs[2].scatter(df['response_rate_control'], df['response_rate_treatment'], color='#4C72B0', alpha=0.7)
        axs[2].set_xlabel('Control response rate', fontsize=14)
        axs[2].set_ylabel('Treatment response rate', fontsize=14)
        axs[2].tick_params(axis='both', labelsize=10)
        axs[2].tick_params(axis='x', labelsize=10)

        # Add a regression line
        import numpy as np
        from scipy.stats import linregress
        x = df['response_rate_control']
        y = df['response_rate_treatment']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # Plot the regression line
        line_x = np.linspace(0, max(x))
        line_y = slope * line_x + intercept
        axs[2].plot(line_x, line_y, color='black', linestyle='--', linewidth=2)

        # Adjust the position of the y-axis label
        axs[2].yaxis.set_label_coords(-0.15, 0.5)

        # Box plot of the response rates
        sns.boxplot(data=[df['response_rate_control'], df['response_rate_treatment']], palette=['#55A868', '#4C72B0'], ax=axs[3])
        axs[3].set_ylabel('Response rate', fontsize=14)
        axs[3].set_title('Distribution of response rates', fontsize=14)
        axs[3].tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)


def uplift_quadrants():
    #st.write(os.listdir())
    s3 = boto3.client('s3')
    bucket_name = 'uplift-model'
    model_uri = 's3://{}/final/Group-By-Project---FourthBrain-/nb/mlruns/517342746135544475/af4b942074eb430c97be548979749e6b/artifacts/class_transformation_model'.format(bucket_name)

# Load the model
    loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

    # model_uri = "nb/mlruns/517342746135544475/af4b942074eb430c97be548979749e6b/artifacts/class_transformation_model"
    # Load the model from the run
    #loaded_model = mlflow.sklearn.load_model(model_uri)

    demo_cols = ['job_blue-collar',
       'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
       'job_self-employed', 'job_services', 'job_student', 'job_technician',
       'job_unemployed', 'marital_divorced', 'marital_married',
       'marital_single', 'education_primary', 'education_secondary',
       'education_tertiary','age_0-25', 'age_25-35', 'age_35-50',
       'age_50-100']
    
    X_test_2 = pd.read_csv('https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/X_test.csv')
    y_test = pd.read_csv('https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/y_test.csv')
    trmnt_test = pd.read_csv('https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/trmnt_test.csv')

    uplift_ct = loaded_model.predict(X_test_2)

    
	# convert the arrays to pandas Series
    uplift_sm = pd.Series(uplift_ct)
    trmnt_test = pd.Series(trmnt_test['treatment_tag'])
    y_test = pd.Series(y_test['conversion'])

    # create a new column in the test set dataframe and assign uplift scores to it
    test_set_df = pd.concat([X_test_2.reset_index(drop=True), trmnt_test.reset_index(drop=True), y_test.reset_index(drop=True),  uplift_sm.reset_index(drop=True)], axis=1)
    test_set_df.columns = list(X_test_2.columns) + ['treatment_tag', 'conversion', 'uplift_score']
    #Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    df = test_set_df
    # Set color palette
    sns.set_palette(['#4C72B0', '#55A868', '#C44E52', '#8172B2'])

    # Categorize customers based on uplift
    df['uplift_category'] = pd.qcut(df['uplift_score'], q=[0, 0.2, 0.5, 0.8, 1], labels=['Lost Causes', 'Sleeping Dogs', 'Persuadable', 'Sure Things'])

    # Plot histogram of uplift by category
    sns.histplot(data=df, x='uplift_score', hue='uplift_category', multiple='stack', bins=30, ax=axs[0, 0])

    # Add annotations to indicate where the sleeping dogs, lost causes, persuadable, and sure things fall
    axs[0, 0].annotate('Sleeping Dogs', xy=(-0.25, 80), xytext=(-0.6, 200))
    axs[0, 0].annotate('Lost Causes', xy=(-0.03, 120), xytext=(-0.2, 250))
    axs[0, 0].annotate('Sure Things', xy=(0.28, 80), xytext=(-0.05, 200) )
    axs[0, 0].annotate('Persuadables', xy=(0.12, 150), xytext=(0.25, 220))

    # Adjust font size of annotations
    axs[0, 0].tick_params(labelsize=10)

    # Create a count plot of the uplift_category column
    sns.countplot(data=df, x='uplift_category', ax=axs[0, 1])

    # Get the counts and percentages for each category
    counts = df['uplift_category'].value_counts()
    counts = counts.sort_index()
    percentages = df['uplift_category'].value_counts(normalize=True) * 100

    # Add annotations for each category with count and percent of total
    for i, category in enumerate(counts.index):
        count = counts[category]
        percent = round(percentages[category], 2)
        axs[0, 1].annotate(f'{count}\n({percent}%)', xy=(i, count), ha='center', va='top', fontsize=10)

    # Set the plot title and axis labels
    axs[0, 1].set_title('Uplift Categories', fontsize=12)
    axs[0, 1].set_xlabel('Uplift Category', fontsize=10)
    axs[0, 1].set_ylabel('Count', fontsize=10)

    # Set the y-axis limit to avoid overlap with the title
    axs[0, 1].set_ylim(0, max(counts.values) * 1.05)

    # Group by uplift category and calculate mean uplift score
    uplift_scores = df.groupby('uplift_category').mean()[['uplift_score']]

    # Create bar plot
    ax = sns.barplot(data=uplift_scores, x=uplift_scores.index, y='uplift_score', ax=axs[1, 0])

    # Add the average uplift score as a label to the bars
    for i, score in enumerate(uplift_scores['uplift_score']):
        ax.text(i, score, round(score, 2), ha='center', va='bottom', fontsize=10)

    #Set the plot title and axis labels

    axs[1, 0].set_title('Average Uplift Score by Uplift Category', fontsize=12)
    axs[1, 0].set_xlabel('Uplift Category', fontsize=10)
    axs[1, 0].set_ylabel('Average Uplift Score', fontsize=10)

    #Plot the decision tree

    target = 'uplift_category'
    rfc = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    rfc.fit(df[demo_cols], df[target])

    tree_to_plot = 0
    plt.figure(figsize=(25, 10))
    plot_tree(rfc.estimators_[tree_to_plot], feature_names=demo_cols, class_names=df[target].cat.categories, filled=True, fontsize=8, max_depth=2, ax=axs[1, 1])

    #Set the plot title and axis labels

    axs[1, 1].set_title('Decision Tree', fontsize=12)
    axs[1, 1].set_xlabel('')
    axs[1, 1].set_ylabel('')

    ##Adjust spacing between plots

    fig.tight_layout(pad=3)

    #Show the plot
    st.pyplot(fig)

    

def app():
    st.set_page_config(page_title='Uplift Model', page_icon=':bar_chart:')
    st.title('Uplift Model - Campaign Analytics')

    tabs = ['Categorical Analysis', 'Numerical Analysis', 'Campaign Results', 'Uplift Segment Results']
    selected_tab = st.sidebar.selectbox('', tabs)

    if selected_tab == 'Categorical Analysis':
        categorical_analysis()
    elif selected_tab == 'Numerical Analysis':
        numerical_analysis()
    elif selected_tab == 'Campaign Results':
        campaign_results()
    elif selected_tab == 'Uplift Segment Results':
        st.write('Uplift Segment Results')
        uplift_quadrants()

if __name__ == '__main__':
    app()
