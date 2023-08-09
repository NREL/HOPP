import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# set to logging.WARNING for fewer messages
logging_level = logging.INFO

# level to print to console
console_level = logging.WARNING


# set up logging to file - see previous section for more details
cluster = os.environ.get('SLURM_CLUSTER_NAME')

formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
if cluster == "eagle":
    print("logging to stdout")
    logging.basicConfig(level=logging_level,
                        datefmt='%m-%d %H:%M',
                        stream=sys.stdout)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
else:
    run_suffix = '_' + datetime.now().isoformat().replace(':', '.')
    log_path = Path.cwd() / "log"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    log_path = log_path / ("hybrid_systems" + run_suffix + ".log")
    print(log_path)
    # logging.basicConfig(level=logging_level,
    #                     datefmt='%m-%d %H:%M',
    #                     filename=str(log_path),
    #                     filemode='w')
    handler = logging.FileHandler(str(log_path))
    handler.setFormatter(formatter)

# define a Handler which writes WARNING messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(console_level)


# Now, define a couple of other loggers which might represent areas in your
# application:

hybrid_logger = logging.getLogger('HybridSim')
flicker_logger = hybrid_logger
bos_logger = hybrid_logger
analysis_logger = hybrid_logger
opt_logger = logging.getLogger('Optimization')

hybrid_logger.addHandler(handler)
opt_logger.addHandler(handler)
hybrid_logger.addHandler(console)
opt_logger.addHandler(console)

logging.getLogger('').propagate = False
logging.getLogger('HybridSim').propagate = False
logging.getLogger('Optimization').propagate = False
