from .record_logger import RecordLogger


class NullRecordLogger(RecordLogger):
    """
    A record logger that simply discards data.
    """
    
    def write(self, data) -> None:
        pass
    
    def flush(self) -> None:
        pass
