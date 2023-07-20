from .a_data_recorder import ADataRecorder
from .null_record_logger import NullRecordLogger
from .record_logger import RecordLogger


class NullDataRecorder(ADataRecorder):
    """
    A DataRecorder that simply discards data.
    """
    
    def __init__(self) -> None:
        pass
    
    def add_columns(self, *column_names) -> None:
        pass
    
    def set_schema(self):
        pass
    
    def accumulate(self, *data, **kwdata) -> None:
        pass
    
    def store(self) -> None:
        pass
    
    def is_setup(self) -> bool:
        return True
    
    def get_column(self, name) -> []:
        return []
    
    def get_record(self, index) -> []:
        return []
    
    def get_records(self) -> []:
        return []
    
    def get_column_map(self) -> {}:
        return {}
    
    def close(self) -> None:
        pass
