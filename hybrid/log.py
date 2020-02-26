import logging
from pathlib import Path

log_path = Path(__file__).parent.parent / "results" / "hybrid_systems.log"

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
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

logger = logging.getLogger('hybrid_system')
logger2 = logging.getLogger('hybrid_analysis')
