import json

import numpy

from .record_logger import RecordLogger


class JSONLinesRecordLogger(RecordLogger):
    """
    Writes data to a JSONLines formatted log file. Each call to write() writes a new JSON object on a new line.
    """
    
    def __init__(self, filename) -> None:
        self._file = open(filename, 'w', encoding='utf-8')
    
    def write(self, data) -> None:
        
        # noinspection PyBroadException
        def object_converter(obj):
            if isinstance(obj, numpy.ndarray):
                # convert numpy arrays to lists
                return obj.tolist()
            elif isinstance(obj, numpy.float32) or \
                    isinstance(obj, numpy.float64) or \
                    isinstance(obj, numpy.int8) or\
                    isinstance(obj, numpy.int16) or \
                    isinstance(obj, numpy.int32) or \
                    isinstance(obj, numpy.int64) or \
                    isinstance(obj, numpy.uint8) or \
                    isinstance(obj, numpy.uint16) or \
                    isinstance(obj, numpy.uint32) or \
                    isinstance(obj, numpy.uint64) or \
                    isinstance(obj, numpy.intp) or \
                    isinstance(obj, numpy.uintp):
                # convert numpy numbers to python numbers
                return obj.item()
            try:
                return obj.toJSON()
            except:
                return obj.__dict__
        
        self._file.write(json.dumps(
            data,
            ensure_ascii=False,
            indent=None,
            separators=(',', ':'),
            default=object_converter))
        self._file.write('\n')
    
    def flush(self) -> None:
        self._file.flush()
    
    def close(self) -> None:
        # noinspection PyBroadException
        try:
            if hasattr(self, 'file') and self._file is not None and not self._file.closed:
                self._file.close()
        except:
            pass
        
  
