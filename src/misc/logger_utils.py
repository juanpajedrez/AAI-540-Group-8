# https://medium.com/swlh/add-log-decorators-to-your-python-project-84094f832181
# decorator

import logging
logger = logging.getLogger('aws')

def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__} with arguments: {args} and keyword arguments: {kwargs}")
        result = func(*args, **kwargs)
        #logger.info(f"Function: {func.__name__} returned: {result}")
        return result
    return wrapper