import logging
from pathlib import Path
from pandas import DataFrame

logger = logging.getLogger('aws')

from sklearn.model_selection import train_test_split

from src.misc.logger_utils import log_function_call

@log_function_call
def split_to_dataset(list_df:
    list[tuple[str, DataFrame]],
    ticker:str, passed_path:Path):
    try:
        for name, df in list_df:
            local_file_path = passed_path / f"{ticker}{name}.csv"
            if not local_file_path.exists():
                df = df.sort_index()
                df.to_csv(local_file_path)
            else:
                continue
    except Exception as e:
        logger.error(e)

@log_function_call
def prod_dev_split(df: DataFrame, ticker: str, full_run: bool):
    try:
        x_dev, x_prod = train_test_split(df, test_size=0.4, shuffle=False)
        dataset = [('x_prod', x_prod)]
        if full_run:
            local_prod_dataset_path = Path().cwd() / "files" / "dataset" / "prod"
            local_prod_dataset_path.mkdir(parents = True, exist_ok = True)
            split_to_dataset(dataset, ticker, local_prod_dataset_path)
        return x_dev
    except Exception as e:
        logger.error(e)


@log_function_call
def dataset_operations(df: DataFrame, ticker: str, y_label_name: str):
    try:
        labels = df[y_label_name]
        features = df.drop(columns=[y_label_name], axis=1)

        x_dev, x_test, y_dev, y_test = train_test_split(labels, features, test_size=0.33, shuffle=False)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

        dataset = [('x_dev', x_dev), ('y_dev', y_dev),
                   ('x_test', x_test), ('y_test', y_test), ('x_val', x_val), ('y_val', y_val)]

        local_dataset_path = Path().cwd() / "files" / "dataset"
        split_to_dataset(dataset, ticker, local_dataset_path)

    except Exception as e:
        logger.error(e)