import streamlit as st
import pandas as pd
import seaborn as sns
from sklift.metrics import uplift_by_percentile
import mlflow
import altair as alt
#import matplotlib.pyplot as plt


CAT_COLS = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'poutcome', 'month', 'age_0-25', 'age_25-35', 'age_35-50', 'age_50-100',
    'days_0-7', 'days_7-14', 'days_14-21', 'days_21-31'
]

NUM_COLS = ['age', 'balance', 'duration', 'pdays', 'previous']

#plt.rcParams['font.family'] = 'serif'
#plt.style.use('seaborn-pastel')


def load_data():
    url_dat = 'https://media.githubusercontent.com/media/toyobam92/Group-By-Project---FourthBrain-/uplift_steps/dat/feature_eng_data.csv'
    return pd.read_csv(url_dat)


import altair as alt


def create_plot(df, col, plot_type):
    if plot_type == 'treatment_tag':
        grouped = (
            df.groupby(['treatment_tag', col])
            .size()
            .reset_index(name='counts')
            .assign(
                percentage=lambda x: x.groupby(['treatment_tag'])['counts']
                .apply(lambda x: x / x.sum() * 100)
            )
            .pivot(index=col, columns=['treatment_tag'], values=['counts', 'percentage'])
        )
        chart = (
            alt.Chart(grouped['percentage'].reset_index())
            .mark_bar()
            .encode(
                x=alt.X(col, title=col),
                y=alt.Y('percentage:Q', title='Percentage'),
                color='treatment_tag:N',
                tooltip=[col, 'percentage']
            )
            .properties(title=f'{col}, treatment with percentage')
            .properties(width=500, height=300)
        )
        return chart

    elif plot_type == 'conversion':
        grouped = (
            df.groupby(['conversion', col])
            .size()
            .reset_index(name='counts')
            .assign(
                percentage=lambda x: x.groupby(['conversion'])['counts']
                .apply(lambda x: x / x.sum() * 100)
            )
            .pivot(index=col, columns=['conversion'], values=['counts', 'percentage'])
        )
        chart = (
            alt.Chart(grouped['percentage'].reset_index())
            .mark_bar()
            .encode(
                x=alt.X(col, title=col),
                y=alt.Y('percentage:Q', title='Percentage'),
                color='conversion:N',
                tooltip=[col, 'percentage']
            )
            .properties(title=f'{col}, conversion with percentage')
            .properties(width=500, height=300)
        )
        return chart

    elif plot_type == 'treatment_tag and conversion':
        grouped = (
            df.groupby(['treatment_tag', 'conversion', col])
            .size()
            .reset_index(name='counts')
            .assign(
                percentage=lambda x: x.groupby(['treatment_tag', 'conversion'])['counts']
                .apply(lambda x: x / x.sum() * 100)
            )
            .pivot(index=col, columns=['treatment_tag', 'conversion'], values=['counts', 'percentage'])
        )
        chart = (
            alt.Chart(grouped['percentage'].reset_index())
            .mark_bar()
            .encode(
                x=alt.X(col, title=col),
                y=alt.Y('percentage:Q', title='Percentage'),
                color=alt.Color('treatment_tag:N'),
                column=alt.Column('conversion:N', title='Conversion'),
                tooltip=[col, 'percentage']
                )
                .properties(title=f'{col}, treatment, and conversion with percentage')
                .properties(width=500, height=300)
        )
        return chart
    
    elif plot_type == 'treatment_tag and conversion numerical':
        group_cols = ['treatment_tag', 'conversion']
        grouped = (
            df.groupby(group_cols)[col]
            .mean()
            .reset_index(name='mean')
            .pivot(index='treatment_tag', columns='conversion', values='mean')
        )
        chart = (
            alt.Chart(grouped.reset_index())
            .mark_bar()
            .encode(
                x=alt.X('treatment_tag:N', title='Treatment Tag'),
                y=alt.Y(['Control', 'Treatment'], title=col),
                color=alt.Color('conversion:N', title='Conversion'),
                tooltip=[col]
            )
            .properties(title=f'{col} vs Treatment Tag and Conversion')
        )
        return chart
    else:
        raise ValueError(f'Invalid plot type: {plot_type}')

def categorical_analysis():
    st.write('Categorical Features')
    data = load_data()
    selected_col = st.selectbox('Select a column', CAT_COLS)

    st.write('Select View to Group Data')
    plot_type = st.selectbox('', ['treatment_tag', 'conversion', 'treatment_tag and conversion'])

    if st.button('Generate Plot'):
        if plot_type:
            chart = create_plot(data, selected_col, plot_type)
            st.altair_chart(chart)
            
def numerical_analysis():
    st.write('Numerical Features')
    data = load_data()
    selected_col = st.selectbox('Select a column', NUM_COLS)

    st.write('Select View to Group Data')
    plot_type = st.selectbox('', ['treatment_tag and conversion numerical'])

    if st.button('Generate Plot'):
        if plot_type:
            chart = create_plot(data, selected_col, plot_type)
            st.altair_chart(chart)


def app():
    st.set_page_config(page_title='Uplift Model', page_icon=':bar_chart:')
    st.title('Uplift Model - Campaign Analytics')
    tabs = ['Categorical Analysis', 'Numerical Analysis', 'Campaign Results', 'Uplift Segment Results']
    selected_tab = st.sidebar.selectbox('', tabs)

# create sidebar widgets for quartile values
#initial_quartile_values = [0, 0.2, 0.5, 0.8, 1]
#quartile_values = [    st.sidebar.number_input(f'Value for {i}% quartile', min_value=0.0, max_value=1.0, value=float(initial_quartile_values[i]), step=0.1) for i in range(5)]
    if selected_tab == 'Categorical Analysis':
        categorical_analysis()
    elif selected_tab == 'Numerical Analysis':
        numerical_analysis()
    #elif selected_tab == 'Campaign Results':
        #campaign_results()
    #elif selected_tab == 'Uplift Segment Results':
        #st.write('Uplift Segment Results')
        #uplift_quadrants(quartile_values, selected_variable)
if __name__ == '__main__':
        app()
