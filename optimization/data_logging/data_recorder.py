from optimization.data_logging.null_record_logger import NullRecordLogger
from optimization.data_logging.record_logger import RecordLogger


class DataRecorder:
    """
    Accumulates data from an experimental run in a tabular format and allows that data to be written out to disk.
    Data is accumulated in a tabular format, and is expected to always match the columns defined.
    """
    
    def __init__(
            self,
            logger: RecordLogger = NullRecordLogger(),
            ) -> None:
        self._is_setup = False
        self._column_map: {any, int} = {}
        self._column_list: [] = []
        self._records: [] = []
        self._record: [] = []
        self._logger: RecordLogger = logger
    
    def __del__(self):
        # noinspection PyBroadException
        try:
            self.close()
        except:
            pass
    
    def add_columns(self, *column_names) -> None:
        """
        Adds columns to the schema. Add columns in the same order you will record them in.
        """
        for column_name in column_names:
            self._column_list.append(column_name)
            self._column_map[column_name] = len(self._column_map)
        if self._is_setup:
            raise Exception("Adding columns after accumulating data is not supported.")
    
    def set_schema(self):
        """
        Call this after all columns have been defined via add_columns().
        Schema changes can only happen before this point.
        Data can only be accumulated after this point.
        """
        self._logger.write_and_flush(self._column_list)
        self._is_setup = True
        self._record = []
    
    def accumulate(self, *data):
        self._record.extend(data)
    
    def store(self) -> None:
        """
        Closes the accumulated record, adds it to self.records and logs it to the logger
        """
        if not self._is_setup:
            raise Exception("Writing data before setting the schema is not supported.")
        
        self._records.append(self._record)
        self._record = []
        self._logger.write_and_flush(self._record)
    
    def is_setup(self) -> bool:
        return self._is_setup
    
    def get_column(self, name) -> []:
        index = self._column_map[name]
        return [record[index] for record in self._records]
    
    def get_record(self, index) -> []:
        return self._records[index]
    
    def get_records(self):
        return self._records
    
    def get_column_map(self):
        return self._column_map
    
    def close(self) -> None:
        """
        Must be called to dispose of an instance
        """
        self._logger.close()
