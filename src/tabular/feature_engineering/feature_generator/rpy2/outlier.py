from rpy2.missing_value import imputing_missing_data
from outliers import smirnov_grubbs as grubbs
from rpy2.robjects.packages import importr
stats = importr('stats')

def imputing_outlier(data, detect_methods='boxplot'):
    '''
    
    :param data: 
    :param detect_methods: boxplot, grubbs, dixon
    :param handle_methods: Multiple_interpolation
    :return: 
    '''
    data.index = range(0,data.shape[0])
    if detect_methods == 'boxplot':
        for col in data.columns:
            QL = stats.quantile(data[[col]], probs = 0.25)
            QU = stats.quantile(data[[col]], probs = 0.75)
            QU_QL = QU - QL
            max_bound = QU + 1.5 * QU_QL
            min_bound = QL - 1.5 * QU_QL
            for index,i in enumerate(data[[col]]):
                if  i > max_bound or i < min_bound:
                    data.loc[index,col] == 'NA'
        return imputing_missing_data(data, methods = 'ppm', target=False)

    if detect_methods == 'grubbs':
        for col in data.columns:
            data_index = grubbs.test(data[[col]], 0.05).index
            outlier_index = [i for in data.index if i not in data_index]
            data.loc[i for in outlier_index, col] == 'NA'
        return imputing_missing_data(data, methods='ppm', target=False)








