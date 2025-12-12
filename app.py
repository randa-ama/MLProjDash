    # app.py (Render-optimized, minimal-change refactor)

import os
import zipfile
import pickle
from functools import lru_cache

import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

# --------------------------
# Dash app init (unchanged)
# --------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server  # for gunicorn: Procfile -> web: gunicorn app:server

# --------------------------
# Lazy loaders (cached)
# --------------------------
@lru_cache(maxsize=1)
def load_df():
    """Load main CSV from zip once, return cleaned dataframe."""
    zip_file_name = "CDC-2019-2021-2023-DATA.csv.zip"
    csv_file_name = "CDC-2019-2021-2023-DATA.csv"
    with zipfile.ZipFile(zip_file_name, mode="r") as archive:
        with archive.open(csv_file_name, mode="r") as csv_file:
            df_local = pd.read_csv(csv_file, low_memory=False)
    # Keep your original cleaning
    df_local = df_local.query("IYEAR != 2024").dropna().drop("Unnamed: 0", axis=1)
    df_local["ADDEPEV3"] = df_local["ADDEPEV3"].replace({"Yes": 1, "No": 0}).astype(float)
    return df_local

@lru_cache(maxsize=1)
def load_lr_data():
    """Load numeric lr csv used in linear models (only when needed)."""
    lr_zip = "CDC-2019-2023-DATA_nums.csv.zip"
    lr_csv = "CDC-2019-2023-DATA_nums.csv"
    with zipfile.ZipFile(lr_zip, mode="r") as archive1:
        with archive1.open(lr_csv, mode="r") as csv_file1:
            lr_data = pd.read_csv(csv_file1, low_memory=False)
    lr_data = lr_data.drop(["Unnamed: 0"], axis=1)
    return lr_data

@lru_cache(maxsize=1)
def load_hac_pickles():
    """Load precomputed HAC pickles (all five)."""
    with open("hac_sil_fig.pkl", "rb") as f:
        hac_sil_fig = pickle.load(f)
    with open("hac_heatmap_fig.pkl", "rb") as f:
        hac_heatmap_fig = pickle.load(f)
    with open("hac_cat_fig.pkl", "rb") as f:
        hac_cat_fig = pickle.load(f)
    with open("fig_mh_bar.pkl", "rb") as f:
        fig_mh_bar = pickle.load(f)
    with open("fig_violin.pkl", "rb") as f:
        fig_violin = pickle.load(f)
    return hac_sil_fig, hac_heatmap_fig, hac_cat_fig, fig_mh_bar, fig_violin

# --------------------------
# Minimal helpers that wrap heavy computations (imports inside)
# --------------------------
def do_logit(X, y, test_size, threshold):
    # Lazy imports so they don't load at startup
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        confusion_matrix,
        roc_curve,
        roc_auc_score,
    )
    import numpy as np
    import pandas as pd

    nums = ["POORHLTH", "MENTHLTH"]
    cats = [
        "IYEAR",
        "BIRTHSEX",
        "ACEDEPRS",
        "DECIDE",
        "DIFFALON",
        "ACEDRINK",
        "ACEDRUGS",
        "ACEPRISN",
        "ACEDIVRC",
        "ACEPUNCH",
        "ACEHURT1",
        "ACESWEAR",
        "ACETOUCH",
        "ACETTHEM",
        "ACEHVSEX",
    ]

    # NOTE: use sparse_output=False for recent sklearn
    preprocess = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first", sparse_output=False), cats),
            ("numeric", "passthrough", nums),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, stratify=y
    )

    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    y_hat_test = (p_test >= threshold).astype(int)

    acc = accuracy_score(y_test, y_hat_test)
    ll = log_loss(y_test, p_test)
    cm = confusion_matrix(y_test, y_hat_test)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)

    logit = pipe.named_steps["model"]
    preprocess_step = pipe.named_steps["preprocess"]

    try:
        feature_names = preprocess_step.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(logit.coef_.shape[1])]

    coefs = logit.coef_.ravel()

    coef_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coeff": np.abs(coefs),
            }
        )
        .sort_values("abs_coeff", ascending=False)
        .head(15)
    )

    return acc, ll, cm, fpr, tpr, auc, coef_df

def do_knn(X, y):
    # Lazy imports
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        roc_curve,
        roc_auc_score,
    )
    import plotly.express as px
    import pandas as pd
    import numpy as np

    nums = ["POORHLTH", "MENTHLTH"]
    cats = [
        "IYEAR",
        "BIRTHSEX",
        "ACEDEPRS",
        "DECIDE",
        "DIFFALON",
        "ACEDRINK",
        "ACEDRUGS",
        "ACEPRISN",
        "ACEDIVRC",
        "ACEPUNCH",
        "ACEHURT1",
        "ACESWEAR",
        "ACETOUCH",
        "ACETTHEM",
        "ACEHVSEX",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first", sparse_output=False), cats),
            ("numeric", "passthrough", nums),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(weights="distance")),
        ]
    )

    param_grid = {"knn__n_neighbors": range(1, 41, 2)}
    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    results_df = pd.DataFrame(grid.cv_results_)
    results_df["k"] = results_df["param_knn__n_neighbors"]
    results_df["mean_score"] = results_df["mean_test_score"]

    best_k = grid.best_params_["knn__n_neighbors"]
    best_score = grid.best_score_

    fig = px.line(
        results_df,
        x="k",
        y="mean_score",
        title=f"Cross-Validated Balanced Accuracy vs. K (best k = {best_k})",
        markers=True,
        labels={
            "k": "Number of Neighbors (k)",
            "mean_score": "Mean CV Balanced Accuracy",
        },
    )

    fig.add_scatter(
        x=[best_k],
        y=[best_score],
        mode="markers+text",
        text=[f"Best k = {best_k}"],
        textposition="top center",
        name="Best k",
    )

    fig.update_layout(hovermode="x unified", showlegend=False)

    # Fit final
    pipe2 = Pipeline(
        [
            ("preprocess", preprocess),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k, weights="distance")),
        ]
    )

    pipe2.fit(X_train, y_train)
    y_pred = pipe2.predict(X_test)
    prob_test = pipe2.predict_proba(X_test)[:, 1]

    knn_acc = accuracy_score(y_test, y_pred)
    knn_bal_acc = balanced_accuracy_score(y_test, y_pred)

    knn_fpr, knn_tpr, _ = roc_curve(y_test, prob_test)
    knn_auc = roc_auc_score(y_test, prob_test)
    knn_roc_fig = px.area(
        x=knn_fpr,
        y=knn_tpr,
        labels=dict(x="False positive rate", y="True positive rate"),
        title=f"ROC Curve (AUC = {knn_auc:.3f})",
    )
    knn_roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    knn_roc_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    # scatter
    knn_scatter = px.scatter(
        X_test.iloc[:10, :],
        x="POORHLTH",
        y="MENTHLTH",
        color=prob_test[:10],
        color_continuous_scale="Magenta",
        symbol=y_test[:10],
        labels={"symbol": "label", "color": "probability"},
        title="KNN Depression Classification Displayed Across Days Depressed and Days Unmotivated",
    )
    knn_scatter.update_traces(marker_size=12, marker_line_width=1.5)
    knn_scatter.update_layout(legend_orientation="h")

    cm = confusion_matrix(y_test, y_pred)
    knn_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="RdPu",
        x=["Predicted: No depression", "Predicted: Yes depression"],
        y=["Actual: No depression", "Actual: Yes depression"],
        labels=dict(x="Predicted label", y="Actual label", color="Count"),
    )
    knn_cm.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    knn_met = html.Ul(
        [
            html.Li(f"Accuracy: {knn_acc:.3f}"),
            html.Li(f"Balanced Accuracy: {knn_bal_acc:.3f}"),
            html.Li(f"AUC: {knn_auc:.3f}"),
        ]
    )

    return fig, knn_roc_fig, knn_met, knn_scatter, knn_cm

# --------------------------
# Precompute-based loading for HAC (lazy)
# --------------------------
def get_hac_figures():
    # loads and returns cached pickles; only called when HAC tab requested
    return load_hac_pickles()

# --------------------------
# Linear regression ACE compute (kept largely as original)
# - moved heavy imports inside function
# --------------------------
def compute_ace_models_and_fig():
    # lazy imports
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import SplineTransformer
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.graph_objects as go

    lr_data = load_lr_data()

    ace_vars = ["ACEDEPRS", "ACESWEAR", "ACETTHEM"]
    num_cols_base = ["AVEDRNK3", "EXEROFT1", "STRENGTH", "PHYSHLTH", "POORHLTH"]
    other_cat_cols = ["IYEAR", "EMPLOY1"]

    ace_metrics = {}
    ace_predictions = {}
    ace_categories_map = {}

    def adjusted_r2_score(y_true, y_pred, n_features):
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    for ace in ace_vars:
        cols_needed = ["MENTHLTH"] + num_cols_base + other_cat_cols + [ace]
        df_ace = lr_data.dropna(subset=cols_needed)

        X = df_ace[num_cols_base + other_cat_cols + [ace]]
        y = df_ace["MENTHLTH"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", sparse_output=False), other_cat_cols + [ace]),
                ("num", "passthrough", num_cols_base),
            ]
        )

        model = Pipeline(
            [
                ("preprocess", preprocess),
                ("spline", SplineTransformer(degree=3, n_knots=8, include_bias=False)),
                ("linreg", LinearRegression()),
            ]
        )

        model.fit(X_train, y_train)

        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        n_features = model.named_steps["preprocess"].transform(X_train).shape[1]

        ace_metrics[ace] = {
            "Train RMSE": mean_squared_error(y_train, yhat_train) ** 0.5,
            "Test RMSE": mean_squared_error(y_test, yhat_test) ** 0.5,
            "Train Adj R2": adjusted_r2_score(y_train, yhat_train, n_features),
            "Test Adj R2": adjusted_r2_score(y_test, yhat_test, n_features),
        }

        categories = sorted(df_ace[ace].unique())
        ace_categories_map[ace] = categories

        base = {col: df_ace[col].mean() for col in num_cols_base}
        for col in other_cat_cols:
            base[col] = df_ace[col].mode()[0]

        plot_df = pd.DataFrame([base.copy() for _ in categories])
        plot_df[ace] = categories
        plot_df = plot_df[num_cols_base + other_cat_cols + [ace]]

        ace_predictions[ace] = model.predict(plot_df)

    # build the figure (same as original)
    bar_color = "#a4a4e3"
    ace_labels_pretty = {
        "ACEDEPRS": "Depressed household member",
        "ACESWEAR": "Verbal abuse as child",
        "ACETTHEM": "Attempted sexual assault",
    }

    fig_ace_local = go.Figure()
    buttons = []

    for i, ace in enumerate(ace_vars):
        x_vals = ace_categories_map[ace]
        y_vals = ace_predictions[ace]
        adj_r2 = ace_metrics[ace]["Test Adj R2"]

        fig_ace_local.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=ace_labels_pretty[ace],
                visible=(i == 0),
                marker_color=bar_color,
            )
        )

        mask = [False] * len(ace_vars)
        mask[i] = True

        buttons.append(
            dict(
                label=ace_labels_pretty[ace],
                method="update",
                args=[
                    {"visible": mask},
                    {
                        "title": (
                            "Cubic Spline Prediction of Bad Mental Health Days<br>"
                            f"<sup>{ace_labels_pretty[ace]} — Test Adjusted R² = {adj_r2:.3f}</sup>"
                        )
                    },
                ],
            )
        )

    first_ace = ace_vars[0]
    first_adj_r2 = ace_metrics[first_ace]["Test Adj R2"]

    fig_ace_local.update_layout(
        title=(
            "Cubic Spline Prediction of Bad Mental Health Days<br>"
            f"<sup>{ace_labels_pretty[first_ace]} — Test Adjusted R² = {first_adj_r2:.3f}</sup>"
        ),
        xaxis_title="ACE Category",
        yaxis_title="Predicted Bad Mental Health Days",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=1.05,
                xanchor="left",
                y=1.0,
                yanchor="top",
                showactive=True,
            )
        ],
    )

    return ace_metrics, fig_ace_local

# compute ACE models lazily when linear tab requested
@lru_cache(maxsize=1)
def load_ace_models_and_fig():
    return compute_ace_models_and_fig()

# --------------------------
# Layout (kept the same)
# --------------------------
app.layout = html.Div(
    [
        html.H1(children="Behavioral Risk Mental Health Dashboard"),
        dcc.Tabs(
            id="tabs",
            value="tab1",
            children=[
                dcc.Tab(label="README: Project Overview", value="tab1"),
                dcc.Tab(label="Data Table", value="tab2"),
                dcc.Tab(label="Models", value="tab3"),
            ],
        ),
        html.Div(id="tabs-content"),
    ]
)

# --------------------------
# Render main tabs (kept unchanged but lazy-load data/pickles)
# --------------------------
@callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == 'tab1':
        return html.Div([
                html.H2('Behavioral Risk Mental Health Dashboard: Predicting Mental Health with Behavioral Risk Factor Variables'),
                html.P('''This app uses behavioral risk variables from 2019, 2021, and 2023 to predict mental health outcomes,
                          focusing primarily on variables relating to adverse childhood experiences, as well as a few other variables.'''),
                
                html.H3('About the Dataset'),
                html.P('''This dataset comes from the CDC\'s Behavioral Risk Factor Surveillance System,
                          a system of comprehensive telephone surveys conducted every year regarding health-related risk behaviors,
                          chronic health conditions, and use of preventative health services for adults in the United States. Each row
                          represents a single respondent with variables including birth sex, year survey was taken, and over 100 behavioral risk
                          related variables. More information can be gathered at this link : https://www.cdc.gov/brfss/about/brfss_faq.htm
                       ''' ),

                html.H3('Target Variables'),
                html.U('Logistic Regression and K Nearest Neighbor'),
                html.P([html.B('ADDEPEV3: '),'''Answer to survey question: (Ever told) (you had) a depressive disorder 
                                                (including depression, major depression, dysthymia, or minor depression)?''']),
                html.U('Linear Regression and Lasso Regularization'),                                
                html.P([html.B('MENTHLTH: '),'''Answer to survey question: Now thinking about your mental health, which includes stress, 
                                                depression, and problems with emotions, for how many days during the past 30 days was your 
                                                mental health not good?''']),                                
                html.H3('Predictor Variables'),
                html.Ul([
                    html.Li([html.B('BIRTHSEX: '),'Assigned sex of respondent at birth']),
                    html.Li([html.B('IYEAR: '), 'Year the respondent took the survey']),
                    html.Li([html.B('POORHLTH: '),'''Answer to survey question: During the past 30 days, for about how many days did poor physical or mental health 
                                                   keep you from doing your usual activities, such as self-care, work, or recreation?''']),
                    html.Li([html.B('MENTHLTH: '), '''Answer to survey question:Now thinking about your mental health, 
                                                     which includes stress, depression, and problems with emotions, 
                                                     for how many days during the past 30 days was your mental health not good?''']),
                    html.Li([html.B('DECIDE: '), '''Answer to survey question: Because of a physical, mental, or emotional condition, 
                                                   do you have serious difficulty concentrating, remembering, or making decisions?''']),
                    html.Li([html.B('DIFFALON: '), '''Answer to survey question: Because of a physical, mental, or emotional condition, 
                                                     do you have difficulty doing errands alone such as visiting a doctor's office or shopping?''']),
                    html.Li([html.B('ACEDEPRS: '), 'Answer to survey question: (As a child) Did you live with anyone who was depressed, mentally ill, or suicidal?']),
                    html.Li([html.B('ACEDRINK: '), 'Answer to survey question: (As a child) Did you live with anyone who was a problem drinker or alcoholic?']),
                    html.Li([html.B('ACEDRUGS: '), 'Answer to survey question: (As a child) Did you live with anyone who used illegal street drugs or who abused prescription medications?']),
                    html.Li([html.B('ACEPRISN: '), 'Answer to survey question: (As a child) Did you live with anyone who served time or was sentenced to serve time in a prison, jail, or other correctional facility?']),
                    html.Li([html.B('ACEDIVRC: '), 'Answer to survey question: (As a child) Were your parents separated or divorced?']),
                    html.Li([html.B('ACEPUNCH: '), 'Answer to survey question: (As a child) How often did your parents or adults in your home ever slap, hit, kick, punch or beat each other up?']),
                    html.Li([html.B('ACEHURT1: '), 'Answer to survey question: (As a child) Not including spanking, (before age 18), how often did a parent or adult in your home ever hit, beat, kick, or physically hurt you in any way?']),
                    html.Li([html.B('ACESWEAR: '), 'Answer to survey question: (As a child) How often did a parent or adult in your home ever swear at you, insult you, or put you down']),
                    html.Li([html.B('ACETOUCH: '), 'Answer to survey question: (As a child) How often did anyone at least 5 years older than you or an adult, ever touch you sexually?']),
                    html.Li([html.B('ACETTHEM: '), 'Answer to survey question: (As a child) How often did anyone at least 5 years older than you or an adult, try to make you touch them sexually?']),
                    html.Li([html.B('ACEHVSEX: '), 'Answer to survey question: (As a child) How often did anyone at least 5 years older than you or an adult, force you to have sex?']),
                    html.Li([html.B('EMPLOY1: '), '''Answer to survey question: Are you currently... (Employed for wages, Self-employed, Out of work for 1 year or more, 
                                                    Out of work for less than 1 year, A homemaker, A student, Retired, or Unable to work)?''']),
                    html.Li([html.B('AVEDRNK3: '), 'Answer to survey question: During the past 30 days, on the days when you drank, about how many drinks did you drink on the average?']),
                    html.Li([html.B('EXEROFT1: '), 'Answer to survey question: How many times per week or per month did you take part in this (physical) activity during the past month?']),
                    html.Li([html.B('STRENGTH: '), 'Answer to survey question: During the past month, how many times per week or per month did you do physical activities or exercises to STRENGTHEN your muscles?']),
                    html.Li([html.B('PHYSHLTH: '), '''Answer to survey question: Now thinking about your physical health, which includes physical illness and injury, 
                                                for how many days during the past 30 days was your physical health not good?'''])
                        ]),

                html.H3('Key Features of Dashboard'),
                html.Ul([
                    html.Li('View rows of the final cleaned dataset in the Data Table tab'),
                    html.Li('See results and interact with variables of the different models in the Models tab')
                        ]),

                html.H3('Instructions for Use'),
                html.Ul([
                    html.Li('Select from K Nearest Neighbor, Logistic Regression, Logistic Regression, Hierarchical Agglomerative Clustering models on the sidebar within the Models tab'),
                    html.Li('Change hyperparameters, such as number of neighbors and train test split, to your liking to view different versions of the Logistic Regression model'),
                    html.Li('Most visualizations are interactive, so hovering over different parts will show more detailed information.'),
                    html.Li('Dropdown menus in visualizations will allow you to visualize different variables'),
                    html.Li('Click on variables in the legend of the violin plots and bar graphs of the Hierarchical Clustering tab to isolate variables to visualize'),
                    html.Li('Note: the KNN tab takes a bit of time to render, so please be patient!')
                        ]),

                html.H3('Authors'),
                html.P('''Randa Ampah, Isabel Delgado, Aysha Hussen, Aniyah McWilliams, 
                          and Jessica Oseghale for the DS 6021 Final Project in the Fall 
                          25 semester of the UVA MSDS program''')

        ])

    if tab == "tab2":
        # load the dataframe only when Data Table tab is requested
        df_local = load_df()
        return html.Div(
            [
                dag.AgGrid(rowData=df_local.to_dict("records"), columnDefs=[{"field": i} for i in df_local.columns])
            ]
        )

    if tab == "tab3":
        return dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavLink("K Nearest Neighbor", href="/models/knn", active="exact"),
                                    dbc.NavLink("Logistic Regression", href="/models/logit", active="exact"),
                                    dbc.NavLink("Multiple Linear Regression w/ Lasso", href="/models/linear", active="exact"),
                                    dbc.NavLink("Hierarchical Clustering", href="/models/hierarchichal", active="exact"),
                                ],
                                vertical=True,
                                pills=True,
                            ),
                            width=2,
                            style={"backgroundColor": "#f8f9fa", "padding": "20px", "height": "100vh"},
                        ),
                        dbc.Col(html.Div(id="sub-tabs-content"), width=10, style={"padding": "40px"}),
                    ]
                ),
                dcc.Location(id="url"),
            ],
        )

    return html.Div("Select a tab.")

# --------------------------
# Sidebar routing (loads HAC pickles lazily)
# --------------------------
@callback(Output("sub-tabs-content", "children"), Input("url", "pathname"))
def update_sidebar_content(pathname):
    # NOTE: when user clicks HAC tab, we load precomputed figures from pickle
    if pathname == "/models/hierarchichal":
        hac_sil_fig_local, hac_heatmap_fig_local, hac_cat_fig_local, fig_mh_bar_local, fig_violin_local = get_hac_figures()
        return html.Div(
            [
                html.H2("Hierarchical Agglomerative Clustering (Gower distance)"),
                dcc.Graph(figure=hac_sil_fig_local),
                dcc.Graph(figure=hac_heatmap_fig_local),
                dcc.Graph(figure=hac_cat_fig_local),
                dcc.Graph(figure=fig_mh_bar_local),
                dcc.Graph(figure=fig_violin_local),
            ]
        )

    elif pathname == "/models/knn":
        # load dataset only when KNN tab requested
        df_local = load_df()
        logit_knn_X = df_local[
            [
                "BIRTHSEX",
                "MENTHLTH",
                "POORHLTH",
                "DECIDE",
                "DIFFALON",
                "IYEAR",
                "ACEDEPRS",
                "ACEDRINK",
                "ACEDRUGS",
                "ACEPRISN",
                "ACEDIVRC",
                "ACEPUNCH",
                "ACEHURT1",
                "ACESWEAR",
                "ACETOUCH",
                "ACETTHEM",
                "ACEHVSEX",
            ]
        ]
        logit_knn_y = df_local["ADDEPEV3"]

        fig, knn_roc_fig, knn_met, knn_scatter, knn_cm = do_knn(logit_knn_X, logit_knn_y)
        return html.Div(
            [
                html.H2("K-Nearest Neighbor Classifier"),
                dcc.Graph(figure=fig),
                html.H3("Model Results"),
                html.H4("Confusion Matrix of Predictions"),
                dcc.Graph(figure=knn_cm),
                html.Div([knn_met]),
                dcc.Graph(figure=knn_roc_fig),
                html.H3("Relevant Graphs"),
                dcc.Graph(figure=knn_scatter),
            ]
        )

    elif pathname == "/models/logit":
        # keep layout same; metrics will be computed in callback update_logit_tab
        return html.Div(
            [
                html.H2("Logistic Regression Model"),
                html.Label("Test set size (%)"),
                dcc.Slider(id="test-size-slider", min=10, max=50, step=5, value=30, marks={i: f"{i}%" for i in range(10,55,5)}),
                html.Br(),
                html.Label("Classification threshold"),
                dcc.Slider(id="threshold-slider", min=0.1, max=0.9, step=0.05, value=0.7),
                html.Br(),
                html.Div(id="logit-metrics"),
                html.H4("Confusion Matrix of Predictions"),
                dcc.Graph(id="logit-confusion"),
                dcc.Graph(id="logit-roc"),
                dcc.Graph(id="logit-coefs"),
            ]
        )

    elif pathname == "/models/linear":
        # compute ACE models lazily and return fig_ace and metrics
        ace_metrics_local, fig_ace_local = load_ace_models_and_fig()
        # Build the same layout you had
        return html.Div(
            [
                html.H2("Multiple Linear Regression (Spline with ACE Variables)"),
                html.P(""),
                html.H3("Lasso Regularization"),
                html.P("Lasso regularization was used to identify the most influential predictors to use within the linear regression model."),
                html.Img(src='/assets/lasso_pic.png', alt='Lasso Results', style={"width": "100%", "height": "auto", "display": "block"}),
                html.H3("MLR Results"),
                dcc.Graph(figure=fig_ace_local),
                html.Br(),
                html.H3("Model Performance Metrics"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr([html.Th("ACE Variable"), html.Th("Train RMSE"), html.Th("Test RMSE"), html.Th("Train Adj R²"), html.Th("Test Adj R²")])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(k),
                                html.Td(f"{ace_metrics_local[k]['Train RMSE']:.3f}"),
                                html.Td(f"{ace_metrics_local[k]['Test RMSE']:.3f}"),
                                html.Td(f"{ace_metrics_local[k]['Train Adj R2']:.3f}"),
                                html.Td(f"{ace_metrics_local[k]['Test Adj R2']:.3f}")
                            ]) for k in ace_metrics_local
                        ])
                    ],
                    style={"width": "70%", "margin": "auto"}
                )
            ]
        )

    return html.Div("Select a model from the sidebar.")

# --------------------------
# Logit metrics callback (left mostly as original)
# --------------------------
@callback(
    Output("logit-metrics", "children"),
    Output("logit-confusion", "figure"),
    Output("logit-roc", "figure"),
    Output("logit-coefs", "figure"),
    Input("test-size-slider", "value"),
    Input("threshold-slider", "value"),
)
def update_logit_tab(test_size_pct, threshold):
    df_local = load_df()
    logit_knn_X = df_local[
        [
            "BIRTHSEX",
            "MENTHLTH",
            "POORHLTH",
            "DECIDE",
            "DIFFALON",
            "IYEAR",
            "ACEDEPRS",
            "ACEDRINK",
            "ACEDRUGS",
            "ACEPRISN",
            "ACEDIVRC",
            "ACEPUNCH",
            "ACEHURT1",
            "ACESWEAR",
            "ACETOUCH",
            "ACETTHEM",
            "ACEHVSEX",
        ]
    ]
    logit_knn_y = df_local["ADDEPEV3"]

    test_size = test_size_pct / 100.0
    acc, ll, cm, fpr, tpr, auc, coef_df = do_logit(logit_knn_X, logit_knn_y, test_size, threshold)

    import plotly.express as px
    import numpy as np

    metrics = html.Ul(
        [
            html.Li(f"Test size: {test_size_pct}%"),
            html.Li(f"Threshold: {threshold:.2f}"),
            html.Li(f"Accuracy: {acc:.3f}"),
            html.Li(f"Log loss: {ll:.3f}"),
            html.Li(f"AUC: {auc:.3f}"),
        ]
    )

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='RdPu',
        x=["Predicted: No depression", "Predicted: Yes depression"],
        y=["Actual: No depression", "Actual: Yes depression"],
        labels=dict(x="Predicted label", y="Actual label", color="Count"),
    )
    cm_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    roc_fig = px.area(
        x=fpr,
        y=tpr,
        labels=dict(x="False positive rate", y="True positive rate"),
        title=f"ROC Curve (AUC = {auc:.3f})",
    )
    roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    roc_fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

    coef_df = coef_df.copy()
    coef_df["direction"] = np.where(
        coef_df["coefficient"] > 0,
        "Risk factors (increase odds of depression)",
        "Protective factors (decrease risk)",
    )

    pretty_map = {
        "encoder__DECIDE_Yes": "Difficulty concentrating: Yes",
        "encoder__ACEDEPRS_Yes": "Lived with depressed adult: Yes",
        "encoder__BIRTHSEX_Male": "Birth sex: Male",
        "encoder__ACEHVSEX_Once": "Forced into sex: Once",
        "encoder__ACEDIVRC_Parents not married": "Parents not married",
        "encoder__ACETOUCH_Never": "Never sexually touched",
        "encoder__ACEPRISN_Yes": "Household member in prison: Yes",
        "encoder__ACEPUNCH_Once": "Witnessed violence: Once",
        "encoder__ACESWEAR_Never": "Never verbally abused",
        "encoder__ACETOUCH_Once": "Sexually touched: Once",
        "encoder__IYEAR_2021": "Survey year: 2021",
        "encoder__DIFFALON_Yes": "Difficulty errands alone: Yes",
        "encoder__ACETTHEM_Once": "Attempted sexual assault: Once",
        "encoder__ACESWEAR_Once": "Verbal abuse: Once",
        "encoder__ACEHVSEX_Never": "Never forced into sex",
    }

    coef_df["pretty_label"] = coef_df["feature"].map(pretty_map).fillna(coef_df["feature"])
    coef_df_sorted = coef_df.sort_values("abs_coeff", ascending=True)

    coef_fig = px.bar(
        coef_df_sorted,
        x="coefficient",
        y="pretty_label",
        color="direction",
        orientation="h",
        title="Top Logistic Regression Coefficients",
        color_discrete_map={
            "Risk factors (increase odds of depression)": "#d81b60",
            "Protective factors (decrease risk)": "#e1bbee",
        },
    )
    coef_fig.update_layout(
        yaxis={"title": "Variable"},
        xaxis={"title": "Coefficient"},
        title=dict(x=0.5, xanchor="center"),
        title_font=dict(size=20),
        margin=dict(l=80, r=60, t=80, b=60),
        legend=dict(title="Effect direction", yanchor="top", y=1.0, xanchor="left", x=1.02),
    )

    return metrics, cm_fig, roc_fig, coef_fig

# --------------------------
# Run server (kept minimal)
# --------------------------
if __name__ == "__main__":
    # When running directly (not via gunicorn), this helps debugging locally.
    app.run_server(host="0.0.0.0", debug=False, port=8051)
