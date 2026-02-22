from pathlib import Path
import logging

# https://coderivers.org/blog/configure-logger-python/
def set_logger(name:str):

    # Let's make sure logs folder exist, else create the folder
    logger_path = Path.cwd() / "logs"
    logger_path.mkdir(parents = True, exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)

    simple_formatter = logging.Formatter("%(name)s:%(levelname)s: %(message)s")
    file_handler = logging.FileHandler('logs/aws.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(simple_formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger