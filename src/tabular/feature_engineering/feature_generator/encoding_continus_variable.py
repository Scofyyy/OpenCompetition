import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler, PowerTransformer
from scipy.special import erfinv
from scipy.stats import kstest, norm


def check_distribution(input_df):
    distribution_types = ["norm", "uniform"]
    p_values = {}

    for distribution_type in distribution_types:
        statistic, p_value = kstest(rvs=input_df, cdf=distribution_type)
        p_values[distribution_type] = p_value

    if max(p_values.values())>= 0.05:
        return "others"

    else:
        if p_values["norm"] >= p_values["uniform"]:
            return "norm"

        else:
            return "uniform"


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
                """
                feature_range : tuple (min, max), default=(0, 1)
                    Desired range of transformed data.
                """
                res = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[[encode_col]])

            elif method == "MinAbs":
                res = MaxAbsScaler().fit_transform(df[[encode_col]])

            elif method == "Normalize":
                """
                norm : 'l1', 'l2', or 'max', optional ('l2' by default)
                    The norm to use to normalize each non zero sample.
                """
                res = Normalizer(norm="l1").fit_transform(df[[encode_col]])

            elif method == "Robust":
                """
                with_centering : boolean, True by default
                    If True, center the data before scaling.
                    This will cause ``transform`` to raise an exception when attempted on
                    sparse matrices, because centering them entails building a dense
                    matrix which in common use cases is likely to be too large to fit in
                    memory.

                with_scaling : boolean, True by default
                    If True, scale the data to interquartile range.

                quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
                    Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
                    Quantile range used to calculate ``scale_``.
                """
                res = RobustScaler(with_centering=True, with_scaling=True,quantile_range=(25.0, 75.0)).fit_transform(df[[encode_col]])

            else:
                raise ValueError(
                    """The column '{}' most likely a uniformly distribution. So, the method value must be in ["MinMax", "MinAbs", "Normalize", "Robust"]""".format(
                        encode_col))

        elif distribution_type == "norm":
            if method == "BoxCox":
                """
                standardize : boolean, default=True
                    Set to True to apply zero-mean, unit-variance normalization to the
                    transformed output.
                """
                res = PowerTransformer(method="box-cox", standardize=True).fit_transform(df[[encode_col]])

            elif method == "Yeo-Johnson":
                res = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(df[[encode_col]])

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
