import sys

version_info = sys.version_info
if version_info.major == 3:
    if version_info.minor == 8:
        from .pyc_files.ProFAST_python38 import ProFAST
    elif version_info.minor == 9:
        from .pyc_files.ProFAST_python39 import ProFAST
    elif version_info.minor == 10:
        from .pyc_files.ProFAST_python310 import ProFAST
    else:
        raise(ImportError(f"ProFAST is not currently available for your python version ({version_info[0]}.{version_info[1]}.{version_info[2]})"))
else:
        raise(ImportError(f"ProFAST is not currently available for your python version ({version_info[0]}.{version_info[1]}.{version_info[2]})"))