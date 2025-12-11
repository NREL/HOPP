import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from hopp import ROOT_DIR

# set up logging to file - see previous section for more details
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
if os.getenv("ENABLE_HOPP_LOGGING",default=False):
    log_level = os.getenv("HOPP_LOG_LEVEL",default="INFO")
    if log_level.upper() == "INFO":
        logging_level = logging.INFO
    if log_level.upper() == "WARNING":
        logging_level = logging.WARNING
    if log_level.upper() == "DEBUG":
        logging_level = logging.DEBUG

    # setup logging to file
    if os.getenv("HOPP_LOG_TO_FILE",default=True):
        run_suffix = '_' + datetime.now().isoformat().replace(':', '.')
        log_path = Path.cwd() / "log"
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_path = log_path / ("hybrid_systems" + run_suffix + ".log")

        logging.basicConfig(level=logging_level,
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=str(log_path),
                        filemode='w')

        handler = logging.FileHandler(str(log_path))
        handler.setFormatter(formatter)
    else:
        # setup logging to console
        logging.basicConfig(level=logging_level,
                            datefmt='%m-%d %H:%M',
                            stream=sys.stdout)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logging_level)
else:
    log_path = ROOT_DIR.parent / "log"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    log_path = log_path / ("empty_log.log")
    handler = logging.FileHandler(str(log_path))
    handler.setFormatter(formatter)

hybrid_logger = logging.getLogger('HybridSim')
flicker_logger = hybrid_logger
bos_logger = hybrid_logger
analysis_logger = hybrid_logger
opt_logger = logging.getLogger('Optimization')

hybrid_logger.addHandler(handler)
opt_logger.addHandler(handler)

logging.getLogger('').propagate = False
logging.getLogger('HybridSim').propagate = False
logging.getLogger('Optimization').propagate = False
