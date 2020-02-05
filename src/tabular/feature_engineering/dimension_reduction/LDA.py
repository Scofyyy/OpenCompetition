import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


def LDA(df, configger):
    """

    Parameters
    ----------
    df: pd.DataFrame. the input DataFrame.
    configger: collections.namedtuple. the configger object like  namedtuple("config",["reduce_col","target_col","n_components"])

    Returns
    -------
    df_t: pd.DataFrame. The result columns named like 'LDA_component_(0,n_components)'
    """
    n_components = configger.n_components
    reduce_col = configger.reduce_col
    target_col = configger.target_col

    if reduce_col is None:
        reduce_col = list(df.columns)
        reduce_col.remove(target_col)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X = df[reduce_col]
    y = df[target_col]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True)
    lda.fit(X_train, y_train)

    res = lda.transform(X=X)
    names = ("LDA_component_" + str(i) for i in range(res.shape[1]))

    res = pd.DataFrame(res, columns=names)
    df_t = pd.concat([df, res], axis=1)

    return df_t
