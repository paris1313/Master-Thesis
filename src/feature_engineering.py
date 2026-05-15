def add_mean_draft(df):

    df['Mean Draft (m)'] = (
        df['DRAFT MP'] +
        df['DRAFT MS'] +
        df['DRAFT FORWARD'] +
        df['DRAFT AFT']
    ) / 4

    return df


def add_course_drift(df):

    df['Course Drift (deg)'] = (
        df['Gyro Heading (deg)'] -
        df['GPS Course (deg)']
    )

    return df
