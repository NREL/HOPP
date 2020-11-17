import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# set to logging.WARNING for fewer messages
logging_level = logging.DEBUG

# set up logging to file - see previous section for more details
cluster = os.environ.get('SLURM_CLUSTER_NAME')

if cluster == "eagle":
    print("logging to stdout")
    logging.basicConfig(level=logging_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        stream=sys.stdout)
else:
    run_suffix = '_' + datetime.now().isoformat().replace(':', '.')
    log_path = Path.cwd() / "log"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    log_path = log_path / ("hybrid_systems" + run_suffix + ".log")
    print(log_path)
    logging.basicConfig(level=logging_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=str(log_path),
                        filemode='w')

# define a Handler which writes WARNING messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

# Now, define a couple of other loggers which might represent areas in your
# application:

hybrid_logger = logging.getLogger('HybridSim')
flicker_logger = logging.getLogger('Flicker')
opt_logger = logging.getLogger('Optimization')
bos_logger = logging.getLogger('BOSCost')
analysis_logger = logging.getLogger('Analysis')

