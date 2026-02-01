from src.misc.logger import set_logger
logger = set_logger('aws')

from src.data.data_local_handler import data_handler_main

def main ():
    # this will act as a glorified cli function caller (which are one off scripts of functions run sequentially)
    data_handler_main(full_exec=False)

if __name__ == '__main__':
    main()