import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import geopandas

from research_data import RawData, ResearchData, raw_data_dir, current_dir

output_dir = current_dir + '/output'
manuscripts_dir = current_dir + '/manuscripts'
raw_data_path = raw_data_dir + '/research_data.plk'
try:
    research_data = pickle.load(open(raw_data_path, 'rb'))
except:
    research_data = ResearchData(cause='anxiety_disorder')
    pickle.dump(research_data, open(raw_data_path, 'wb'))

COLORS = ['#4e79a7', '#f28e2b', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ac', '#e15759']

plt.rcParams.update({'font.size': 8})
plt.rcParams['figure.constrained_layout.use'] = True

import matplotlib.pyplot as plt


def save_figure(title):
    plt.tight_layout()
    plt.savefig(
        manuscripts_dir + f"/{title}.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_geo_distribution(data, variable, year, ax):
    basemap = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    basemap['name'] = RawData.normalize_countries(basemap['name'])

    var = data[data['year'] == year][[research_data.country_key, variable]]
    merged_map = basemap.merge(var, left_on='name', right_on='country', how='left')
    merged_map.plot(
        column=variable,
        legend=True,
        cmap='coolwarm',
        figsize=(10, 4),
        missing_kwds=dict(color="lightgrey"),
        ax=ax
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_title(f"{variable.replace('_', ' ')} in {year}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    return ax


if True:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.4)

    for idx, f in enumerate(research_data.burden_cols):
        ax = axs[idx]
        plot_geo_distribution(research_data.data, f, year=2019, ax=ax)

    fig.suptitle('Trend of Anxiety Disorder Burden', fontsize=12)
    save_figure('Figure 1')


def calculate_trends(data, variable, ret: Literal['time_effect', 'country_effect'] = 'time_effect'):
    data = data[[research_data.year_key, variable]].dropna()

    if data.empty:
        return None

    y = np.log(data[variable])
    X = sm.add_constant(data[research_data.year_key])

    ols_model = sm.OLS(endog=y, exog=X).fit()

    target_param = {
        'time_effect': research_data.year_key,
        'country_effect': 'const'
    }.get(ret, lambda a: a)

    beta_mean = ols_model.params[target_param]
    beta_range = ols_model.conf_int().loc[target_param].values
    beta = np.insert(beta_range, 1, beta_mean)

    g = 100 * (np.exp(beta) - 1)

    return_param = {
        'time_effect': g,
        'country_effect': beta
    }.get(ret, lambda a: a)

    return pd.Series(return_param, ['lower', 'mean', 'upper'])


def plot_growth_trends(data, variable, ax):
    res = data.groupby([research_data.country_key])[[research_data.year_key, variable]].apply(
        calculate_trends, variable=variable
    )

    res['category'] = res.groupby([research_data.country_key])[['lower', 'mean', 'upper']].apply(
        lambda a:
        'increase' if (a.values > 0).all()
        else 'decrease' if (a.values < 0).all()
        else 'no sign'
    )

    res[research_data.category_key] = data.groupby([research_data.country_key])[research_data.category_key].apply(
        lambda a: a.unique()[0])
    res[research_data.category_key] = pd.Categorical(
        res[research_data.category_key],
        categories=research_data.countries_categories.keys(),
        ordered=True
    )
    res = res.sort_values(by=research_data.category_key, ascending=False).dropna()

    sns.scatterplot(
        data=res,
        y=res.index,
        x='mean',
        hue='category',
        ax=ax, palette={'increase': '#2ecc71', 'decrease': '#e74c3c', 'no sign': '#3498db'}
    )
    ax.hlines(y=res.index, xmin=res.lower, xmax=res.upper, color='grey', alpha=0.4)
    ax.vlines(
        ymin=res[res[research_data.category_key] == res[research_data.category_key].min()].index.max(),
        ymax=res[res[research_data.category_key] == res[research_data.category_key].max()].index.min(),
        x=0,
        linestyles='--', colors='red', alpha=0.4
    )
    for group in res[research_data.category_key].unique():
        subset_res = res[res[research_data.category_key] == group]
        ax.fill_betweenx(
            y=subset_res.index,
            x1=res['lower'].min(),
            x2=res['upper'].max(),
            alpha=0.2,
            label=group
        )

    ax.set_ylabel("countries")
    ax.get_legend().remove()

    title = f"{variable.replace('_', ' ')}"
    ax.set_title(title)

    return ax


if True:
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

    for idx, f in enumerate(research_data.burden_cols):
        ax = axs[idx]

        plot_growth_trends(research_data.data, f, ax=ax)

        if idx == 2:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig.suptitle('Burden Growth Trends', fontsize=12)
    save_figure('Figure 2')

if True:
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

    for idx, f in enumerate(research_data.housing_cols):
        ax = axs[idx]

        plot_growth_trends(research_data.data, f, ax=ax)

        if idx == 2:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig.suptitle('Housing Affordability Growth Trends', fontsize=12)
    save_figure('Figure 3')

if True:

    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import make_interp_spline


    def plot_association(
            data,
            x,
            y,
            years
    ):
        year_colors = {year: color for year, color in zip(years, COLORS[:len(years)])}

        res = pd.DataFrame()
        for idx, t in enumerate(years):
            t_data = data[data[research_data.year_key] == t].sort_values(by=x)

            lowess_series = lowess(
                endog=t_data[y],
                exog=t_data[x],
                frac=4 / 5
            )

            lowess_x = lowess_series[:, 0]
            lowess_y = lowess_series[:, 1]
            t_data = t_data.dropna()
            bspline = make_interp_spline(lowess_x, lowess_y)
            xnew = np.linspace(t_data[x].min(), t_data[x].max(), 50)
            ynew = bspline(xnew)

            t_res = pd.DataFrame({
                x: xnew,
                y: ynew,
                research_data.year_key: [t] * len(xnew)
            })

            res = pd.concat([res, t_res], ignore_index=True)

        sns.lineplot(
            data=res, x=x, y=y, hue=research_data.year_key,
            linestyle='--', ax=ax, palette=year_colors
        )
        sns.scatterplot(
            data=data[data.year.isin(years)],
            x=x,
            y=y,
            hue=research_data.year_key,
            # style=research_data.category_key,
            ax=ax,
            palette=year_colors
        )

        title = f"Association of {x.replace('_', ' ')} and {y.replace('_', ' ')}"
        ax.legend(title='Year', loc='upper left')
        ax.set_title(title)
        ax.set_xlabel(x.replace('_', ' '))
        ax.set_ylabel(y.replace('_', ' '))

        return ax


    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    for idx, y in enumerate(research_data.housing_cols[1:]):
        ax = axs[idx]

        plot_association(
            data=research_data.data,
            x=y,
            y='anxiety_disorders_Incidence_rate',
            years=[2010, 2015, 2019]
        )

    save_figure('Figure 4')

if False:
    full_data = research_data.data

    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.iolib.summary2 import summary_col


    def export_regression_results(
            countries_group: Literal['OECD_countries', 'high_countries', 'non_high_countries'],
            disease: Literal['mental', 'depressive', 'anxiety']
    ):

        X_cols = [
            'year',
            'standardised_rent_income_ratio',
            'standardised_price_income_ratio',
            'socio_demographic_index',
            'psychologists',
            'audiologists_and_counsellors'
        ]

        # if countries_group == 'high_countries':
        #     group = research_data.high_sdi_countries
        #
        # elif countries_group == 'non_high_countries':
        #     group = research_data.non_high_sdi_countries
        #
        # else:
        #     group = research_data.countries

        group = research_data.countries_categories[countries_group]

        group_data = full_data[full_data[research_data.country_key].isin(group)].copy()

        output = []
        for y_col in research_data.burden_cols:

            if ('YLLs' in y_col) or (disease not in y_col):
                continue

            temp_data = group_data[np.append([research_data.country_key, y_col], X_cols)].dropna()

            if temp_data.empty:
                continue

            temp_y = temp_data[y_col].rename(y_col.replace('_', ' '))
            temp_X = temp_data[X_cols].rename(columns={
                original: original.replace('_', ' ')
                for original in X_cols
            })

            model = MixedLM(
                endog=temp_y,
                exog=temp_X,
                groups=temp_data[research_data.country_key],
            ).fit(reml=True)

            output += [model]

        if len(output) > 0:
            results_table = summary_col(
                output,
                stars=True,

            )

            results_table.tables[0].iloc[::2].to_csv(
                output_dir +
                f"/regression results {disease} {countries_group.replace('_', ' ')}.csv"
            )

            print(
                results_table
            )


    # for group in ['OECD_countries', 'high_countries', 'low_countries']:
    for group in research_data.countries_categories.keys():
        export_regression_results(group, disease='anxiety')

print()
