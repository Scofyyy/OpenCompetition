import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler, PowerTransformer, StandardScaler
from scipy.special import erfinv
from scipy.stats import kstest, norm


def check_distribution(input_df):
    distribution_types = ["norm", "uniform"]
    for distribution_type in distribution_types:
        statistic, p_value = kstest(input_df, distribution_type)
        if p_value >= 0.05:
            return distribution_type

    return "others"


class GaussRankScaler():
    def __init__(self):
        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform(self, X):
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        j_range = len(j) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv(transformed)

        return transformed


def encode_continuous_variable(df, configgers):
    """
    encoder of continuous variables ,and the method can be in ["MinMax","MinAbs","Normalize","Robust","BoxCox","Yeo-Johnson","RankGuass","icdf"]
    Parameters
    ----------
    df: pd.DataFrame, the input dataframe.
    configgers: list of namedtuple, the config setting of encoding continuous variables like namedtuple("config",["encode_col","method"])

    Returns
    -------
    df_t: pd.DataFrame, the DataFrame after transform, the result columns after transform named like "{origin_column_name}_{method}"
    """
    df_t = df

    for configger in configgers:
        encode_col = configger.encode_col
        method = configger.method

        distribution_type = check_distribution(df[[encode_col]])
        if distribution_type == "uniform":
            if method == "MinMax":
                res = MinMaxScaler().fit_transform(df[[encode_col]])

            elif method == "MinAbs":
                res = MaxAbsScaler().fit_transform(df[[encode_col]])

            elif method == "Normalize":
                res = Normalizer().fit_transform(df[[encode_col]])

            elif method == "Robust":
                res = RobustScaler().fit_transform(df[[encode_col]])

            else:
                raise ValueError(
                    """The column '{}' most likely a uniformly distribution. So, the method value must be in ["MinMax", "MinAbs", "Normalize", "Robust"]""".format(
                        encode_col))

        elif distribution_type == "norm":
            if method == "BoxCox":
                res = PowerTransformer(method="box-cox").fit_transform(df[[encode_col]])

            elif method == "Yeo-Johnson":
                res = PowerTransformer(method="yeo-johnson").fit_transform(df[[encode_col]])

            elif method == "RankGuass":
                res = GaussRankScaler().fit_transform(df[[encode_col]])
            else:
                raise ValueError(
                    """The column '{}' most likely a normal distribution. So, the method value must be in ["BoxCox", "Yeo-Johnson", "Normalize"]""".format(
                        encode_col))

        else:
            if method == "ICDF":
                res = norm.ppf(df[[encode_col]])
            else:
                raise ValueError(
                    """The column '{}' maybe is others distribution. So, the method value must be in ["BoxCox", "Yeo-Johnson", "Normalize"]""".format(
                        encode_col))

        df_t.loc[:, "_".join([encode_col, method])] = res

    return df_t
