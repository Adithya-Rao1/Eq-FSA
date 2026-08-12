import logging
from datetime import datetime

# Generate a filename like 'log_20240324_153045.log'
log_filename = datetime.now().strftime('log_%Y%m%d_%H%M%S.log')

logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logging.getLogger('boto3').propagate = False
logging.getLogger('botocore').propagate = False
logging.getLogger('urllib3.connectionpool').propagate = False
logging.getLogger('matplotlib').propagate = False
logging.getLogger('s3transfer').propagate = False