import numpy as np
from scipy import stats


def remove_outliers(df, columns, threshold=2.7):

    for col in columns:

        z_scores = np.abs(stats.zscore(df[col]))

        df = df[z_scores < threshold]

    return df


def filter_operational_conditions(df):

    df = df.loc[
        df['Voyage Status'].str.contains('Sailing') &
        (df['M/E FLOW METER GROUP ALARM'] != 1) &
        (df['SOW (knots)'] > 5) &
        (df['M/E RPM'] > 0) &
        (df['Depth (m)'] > 70)
    ]

    return df
