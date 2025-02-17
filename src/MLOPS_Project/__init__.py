import logging
import os
import sys

logging_str = "[%(asctime)s %(levelname)s: %(module)s: %(message)s]"
# levelname is like INFO, DEBUG, ERROR and WARN

log_dir = "logs"
log_filepath = os.path.join(log_dir, "logging.log")

if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_filepath)],
    ## streamhandler to see the logs in ther terminal at real time
    ## filehandler to write logs to a file
)

logger = logging.getLogger("MLOPS_Logger")
