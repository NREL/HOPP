import os
from collections import OrderedDict

from .a_data_recorder import ADataRecorder
from .JSON_lines_record_logger import JSONLinesRecordLogger, RecordLogger


class TableDataRecorder(ADataRecorder):
    """
    Accumulates data while keeping track of which entries already exist, to not overwrite
    """

    def __init__(self, log_filename):
        self._logger: RecordLogger = JSONLinesRecordLogger(str(log_filename))
        self._is_setup = False
        self._column_map: {any, int} = {}
        self._column_list: [] = []
        self._records = []
        self._record = []
        self._record_index = 0
        self._record_keys = OrderedDict()
        self._key_cols = []

    def __del__(self):
        # noinspection PyBroadException
        try:
            self._logger.close()
        except:
            pass

    def add_columns(self, *column_names) -> None:
        for column_name in column_names:
            self._column_list.append(column_name)
            self._column_map[column_name] = len(self._column_map)
        if self._is_setup:
            raise Exception("Adding columns after accumulating data is not supported.")
        self.set_schema()

    def set_schema(self):
        self._logger.write_and_flush(self._column_list)
        self._is_setup = True
        self._initialize_record()

    def set_index_columns(self, index_cols):
        for i in index_cols:
            ind = self._column_map[i]
            self._key_cols.append(ind)

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

        index = []
        for i in self._key_cols:
            index.append(self._record[i])

        if tuple(index) in self._record_keys.keys():
            return
        self._records.append(self._record)
        self._logger.write_and_flush(self._record)
        self._initialize_record()

    def _initialize_record(self) -> None:
        self._record = [None] * len(self._column_list)
        self._record_index = 0
