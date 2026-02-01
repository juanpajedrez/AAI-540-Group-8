import logging

from pandas import DataFrame

logger = logging.getLogger('aws')

from sklearn.model_selection import train_test_split

from src.misc.logger_utils import log_function_call

@log_function_call
def split_to_dataset(list_df: list[tuple[str, DataFrame]], ticker:str):
    try:
        for name, df in list_df:
            df.to_csv(f'files/dataset/{ticker}{name}.csv')
    except Exception as e:
        logger.error(e)

@log_function_call
def dataset_operations(df: DataFrame, ticker: str, y_label_name: str):
    try:
        labels = df[y_label_name]
        features = df.drop(columns=[y_label_name], axis=1)

        x_dev, x_prod, y_dev, y_prod = train_test_split(features, labels, test_size=0.4, random_state=12)
        x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.33, random_state=12)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=12)

        dataset = [('x_prod', x_prod), ('y_prod', y_prod), ('x_dev', x_dev), ('y_dev', y_dev),
                   ('x_test', x_test), ('y_test', y_test), ('x_val', x_val), ('y_val', y_val)]

        split_to_dataset(dataset, ticker)

    except Exception as e:
        logger.error(e)