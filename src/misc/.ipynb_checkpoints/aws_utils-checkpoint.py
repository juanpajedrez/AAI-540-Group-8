import logging
logger = logging.getLogger('aws')

# NOTE, THIS WORKS in sagemaker jupyter notebooks, working on setting the .yaml config file
# to add AWS secret key id, AWS access key, or AWS user that reads data from the S3 bucket only
import os
import boto3
from sagemaker.core.helper.session_helper import get_execution_role, Session
from src.misc.logger_utils import log_function_call
from typing import Tuple

@log_function_call
def setup_aws_sagemaker_resources() -> Tuple[Session, str, str, str]:
    try:
        sess = Session()
        region = sess.boto_region_name
        bucket = sess.default_bucket()
        role = get_execution_role()
        return sess, region, bucket
    except Exception as e:
        logger.error(e)