"""
GBD Results tool:
Use the following to cite data included in this download:
Global Burden of Disease Collaborative Network.
Global Burden of Disease Study 2019 (GBD 2019) Results.
Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.
Available from https://vizhub.healthdata.org/gbd-results/.
"""
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from os.path import dirname, realpath
from typing import Literal
import country_converter
import pickle

current_dir = dirname(realpath(__file__))
raw_data_dir = current_dir + '/raw_data'


@dataclass
class RawData:
    country_key = 'country'
    year_key = 'year'
    keys = [country_key, year_key]

    @abstractmethod
    def load(self, *args, **kwargs):
        ...

    def __post_init__(self):
        data = self.load()
        data[self.country_key] = self.normalize_countries(data[self.country_key])
        self.data = data

    @staticmethod
    def normalize_countries(raw_countries: pd.Series):
        unique_countries = raw_countries.unique()

        normalized_countries = country_converter.convert(
            unique_countries, to='name_short', not_found=None
        )

        reference_dict = {
            unique_countries[i]: normalized_countries[i]
            for i in range(len(unique_countries))
        }

        return raw_countries.apply(lambda c: reference_dict.get(c))


@dataclass
class MentalHealthData(RawData):
    cause: Literal['mental disorder', 'anxiety disorder', 'depressive disorder']

    def load(self):
        raw_data = pd.read_csv(raw_data_dir + '/IHME-GBD_2019_MHGBD_Both_Gender.csv')
        raw_data['measure_name'] = raw_data['measure_name'].apply(lambda a: a.split(' ')[0])
        raw_data['cause_name'] = raw_data['cause_name'].apply(lambda a: a.lower().replace(' ', '_'))

        processed_data = pd.DataFrame()
        for measure_name in raw_data['measure_name'].unique():
            for cause_name in raw_data['cause_name'].unique():

                if self.cause not in cause_name:
                    continue

                temp_data = \
                    raw_data[
                        (raw_data.measure_name == measure_name) &
                        (raw_data.cause_name == cause_name)
                        ][['location_name', 'year', 'val']].rename(
                        columns={
                            'location_name': self.country_key,
                            'val': f'{cause_name}_{measure_name}_rate'
                        }
                    )

                if temp_data.empty:
                    continue

                if processed_data.empty:
                    processed_data = temp_data
                else:
                    processed_data = pd.merge(
                        processed_data,
                        temp_data,
                        on=self.keys
                    )

        return processed_data

    def __post_init__(self):
        super().__post_init__()


"""
https://data.oecd.org/price/housing-prices.htm#indicator-chart
OECD (2023), Housing prices (indicator). doi: 10.1787/63008438-en (Accessed on 05 August 2023)
"""


class HousingData(RawData):

    def load(self, *args, **kwargs):
        raw_data = pd.read_csv(raw_data_dir + '/OECD_House_Prices.csv')
        raw_data = raw_data[
            raw_data.TIME.apply(lambda a: False if 'Q' in a else True)
        ].reset_index(drop=True)

        processed_data = pd.DataFrame()
        for i in raw_data.Indicator.unique():

            # if i not in ['Standardised price-income ratio', 'Standardised price-rent ratio']:
            #     continue

            rename_dict = {
                'Country': self.country_key,
                'TIME': self.year_key,
                'Value': i.lower().replace(', s.a.', '').replace(' ', '_').replace('-', '_')
            }

            temp = raw_data[raw_data.Indicator == i][rename_dict.keys()].rename(
                columns=rename_dict
            )

            if processed_data.empty:
                processed_data = temp
            else:
                processed_data = pd.merge(
                    processed_data,
                    temp,
                    on=self.keys,
                    how='outer'
                )

        processed_data[self.year_key] = processed_data[self.year_key].astype(int)

        processed_data['standardised_rent_income_ratio'] = \
            processed_data['standardised_price_income_ratio'] / processed_data['standardised_price_rent_ratio']

        return processed_data[self.keys + ['nominal_house_price_indices', 'standardised_price_income_ratio', 'standardised_rent_income_ratio']]

    def __post_init__(self):
        super().__post_init__()


class SDIData(RawData):

    def load(self, *args, **kwargs):
        raw_data = pd.read_csv(raw_data_dir + '/IHME-GBD_2019_SDI.csv', encoding_errors='ignore')

        processed_data = pd.DataFrame()
        for idx in np.arange(len(raw_data)):
            row = raw_data.loc[idx, :].reset_index()

            c = row.iloc[0, 1]

            temp_data = pd.DataFrame(
                dict(
                    year=row.iloc[1:, 0].astype(int),
                    socio_demographic_index=row.iloc[1:, 1].astype(float)
                ),
            )
            temp_data[self.country_key] = c
            processed_data = pd.concat([processed_data, temp_data], ignore_index=True)

        return processed_data

    def __post_init__(self):
        super().__post_init__()


class HealthWorkerData(RawData):

    def load(self, *args, **kwargs):
        raw_data = pd.read_csv(raw_data_dir + '/IHME-GBD_2019_Health_Worker_Density.csv')

        processed_data = pd.DataFrame()
        for cate in ['Psychologists', 'Audiologists and Counsellors']:
            rename_dict = {
                'location_name': self.country_key,
                'year_id': self.year_key,
                'mean': cate.lower().replace(' ', '_')
            }

            temp = raw_data[raw_data.cadre == cate][rename_dict.keys()].rename(
                columns=rename_dict
            )

            if processed_data.empty:
                processed_data = temp
            else:
                processed_data = pd.merge(processed_data, temp, on=self.keys)

        return processed_data

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ResearchData:
    cause: Literal['mental disorder', 'anxiety disorder', 'depressive disorder']

    country_key = 'country'
    year_key = 'year'
    category_key = 'SDI_group'
    keys = [country_key, year_key]

    def __post_init__(self):

        self.mental_health_data = MentalHealthData(cause=self.cause).data
        self.housing_data = HousingData().data
        self.sdi_data = SDIData().data
        self.health_worker_data = HealthWorkerData().data

        self.data = self.auto_merge()
        self.countries_categories = self.categorize()

        self.burden_cols = self.mental_health_data.columns.difference(self.keys)
        self.housing_cols = self.housing_data.columns.difference(self.keys)
        self.control_cols = np.append(
            self.sdi_data.columns.difference(self.keys),
            self.health_worker_data.columns.difference(self.keys)
        )

    def auto_merge(self):

        data = pd.DataFrame()
        for attr, df in self.__dict__.items():
            if isinstance(df, pd.DataFrame):

                self.__setattr__(attr, df)

                if data.empty:
                    data = df
                else:
                    data = pd.merge(data, df, on=self.keys)

        return data

    def categorize(self, start_year=2015):

        countries_categories = {}

        sdi_quintiles = pd.read_csv(raw_data_dir + '/IHME_GBD_2019_SDI_QUINTILES.csv')
        mean_sdi = self.data[self.data[self.year_key] >= start_year].groupby(self.country_key)[
            'socio_demographic_index'].mean()
        for idx, row in sdi_quintiles.iterrows():
            countries = mean_sdi[mean_sdi.between(row.lower_bound, row.upper_bound)].index.values
            self.data.loc[self.data[self.country_key].isin(countries), self.category_key] = row.sdi_quintile
            countries_categories[row.sdi_quintile] = countries

        self.data[self.category_key] = pd.Categorical(
            self.data[self.category_key],
            categories=sdi_quintiles['sdi_quintile'],
            ordered=True
        )

        return countries_categories

    @property
    def high_sdi_countries(self):
        return self.data[self.data[self.category_key] == 'High SDI'][self.country_key].unique()

    @property
    def non_high_sdi_countries(self):
        return self.data[self.data[self.category_key] != 'High SDI'][self.country_key].unique()

    @property
    def years(self):
        return self.data[self.country_key].unique()

    @property
    def countries(self):
        return self.data[self.country_key].unique()
