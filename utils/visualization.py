import plotly.express as px
import plotly.graph_objects as go

def plot_survival_distribution(df):
    if 'Survived' not in df.columns: return None
    counts = df['Survived'].value_counts().reset_index()
    counts.columns = ['Survived', 'Count']
    counts['Survived'] = counts['Survived'].map({0: 'Not Survived', 1: 'Survived'})
    fig = px.pie(counts, values='Count', names='Survived', title='Survival Distribution',
                 color='Survived', color_discrete_map={'Not Survived': '#EF553B', 'Survived': '#00CC96'}, hole=0.4)
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig

def plot_age_distribution(df):
    if 'Age' not in df.columns: return None
    fig = px.histogram(df, x='Age', nbins=30, title='Age Distribution', color_discrete_sequence=['#AB63FA'])
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig

def plot_fare_distribution(df):
    if 'Fare' not in df.columns: return None
    fig = px.histogram(df, x='Fare', nbins=30, title='Fare Distribution', color_discrete_sequence=['#FFA15A'])
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig

def plot_gender_vs_survival(df):
    if 'Sex' not in df.columns or 'Survived' not in df.columns: return None
    data = df.copy()
    data['Survived'] = data['Survived'].map({0: 'Not Survived', 1: 'Survived'})
    fig = px.histogram(data, x='Sex', color='Survived', barmode='group', title='Gender vs Survival',
                       color_discrete_map={'Not Survived': '#EF553B', 'Survived': '#00CC96'})
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig

def plot_pclass_vs_survival(df):
    if 'Pclass' not in df.columns or 'Survived' not in df.columns: return None
    data = df.copy()
    data['Survived'] = data['Survived'].map({0: 'Not Survived', 1: 'Survived'})
    data['Pclass'] = data['Pclass'].astype(str)
    fig = px.histogram(data, x='Pclass', color='Survived', barmode='group', title='Passenger Class vs Survival',
                       category_orders={"Pclass": ["1", "2", "3"]},
                       color_discrete_map={'Not Survived': '#EF553B', 'Survived': '#00CC96'})
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig

def plot_embarked_vs_survival(df):
    if 'Embarked' not in df.columns or 'Survived' not in df.columns: return None
    data = df.copy()
    data['Survived'] = data['Survived'].map({0: 'Not Survived', 1: 'Survived'})
    fig = px.histogram(data, x='Embarked', color='Survived', barmode='group', title='Embarked vs Survival',
                       color_discrete_map={'Not Survived': '#EF553B', 'Survived': '#00CC96'})
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty: return None
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title='Correlation Heatmap', color_continuous_scale='RdBu_r')
    fig.update_layout(template='plotly_dark', margin=dict(t=40, b=0, l=0, r=0))
    return fig
