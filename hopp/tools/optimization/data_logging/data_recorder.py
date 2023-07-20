import os

from .JSON_lines_record_logger import JSONLinesRecordLogger
from .a_data_recorder import ADataRecorder
from .null_record_logger import NullRecordLogger
from .record_logger import RecordLogger


class DataRecorder(ADataRecorder):
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
        self._record_index: int = 0
        self._logger: RecordLogger = logger
    
    def __del__(self):
        # noinspection PyBroadException
        try:
            self.close()
        except:
            pass
    
    def add_columns(self, *column_names) -> None:
        for column_name in column_names:
            self._column_list.append(column_name)
            self._column_map[column_name] = len(self._column_map)
        if self._is_setup:
            raise Exception("Adding columns after accumulating data is not supported.")
    
    def set_schema(self):
        self._logger.write_and_flush(self._column_list)
        self._is_setup = True
        self._initialize_record()
    
    def accumulate(self, *data, **kwdata) -> None:
        # accumulate data list
        num_data = len(data)
        for i in range(num_data):
            self._record[self._record_index + i] = data[i]
        self._record_index += num_data
        
        # accumulate keyword data
        for key, value in kwdata.items():
            self._record[self._column_map[key]] = value
    
    def store(self) -> None:
        if not self._is_setup:
            raise Exception("Writing data before setting the schema is not supported.")
        
        self._records.append(self._record)
        self._logger.write_and_flush(self._record)
        self._initialize_record()
    
    def is_setup(self) -> bool:
        return self._is_setup
    
    def get_column(self, name) -> []:
        index = self._column_map[name]
        return [record[index] for record in self._records]
    
    def get_record(self, index) -> []:
        return self._records[index]
    
    def get_records(self) -> []:
        return self._records
    
    def get_column_map(self) -> {}:
        return self._column_map
    
    def close(self) -> None:
        self._logger.close()
    
    def _initialize_record(self) -> None:
        self._record = [None] * len(self._column_list)
        self._record_index = 0
    
    @staticmethod
    def make_data_recorder(output_path: str, log_name: str = 'log') -> 'DataRecorder':
        '''
        Makes a JSONLinesRecordLogger based DataRecorder for logging this run
        :param log_name: the what to name this file (has .jsonl appended to it)
        :return: a DataRecorder for this run
        '''
        
        # log_filename = os.path.join(output_path, run_name + '_log' + '.jsonl')
        log_filename = os.path.join(output_path, log_name + '.jsonl')
        return DataRecorder(JSONLinesRecordLogger(log_filename))
