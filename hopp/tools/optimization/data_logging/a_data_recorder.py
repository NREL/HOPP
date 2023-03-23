from abc import abstractmethod


class ADataRecorder:
    """
    Abstract class defining an interface for accumulating data from an experimental run in a tabular format and
    possibly writing that data out to disk. Data is accumulated in a tabular format, and is expected to always match
    the columns defined.
    """
    
    @abstractmethod
    def add_columns(self, *column_names) -> None:
        """
        Adds columns to the schema. Add columns in the same order you will record them in.
        """
        pass
    
    @abstractmethod
    def set_schema(self) -> None:
        """
        Call this after all columns have been defined via add_columns().
        Schema changes can only happen before this point.
        Data can only be accumulated after this point.
        """
        pass
    
    @abstractmethod
    def accumulate(self, *data, **kwdata) -> None:
        """
        Accumulates data into the recorder.
        Data must be either accumulated in the same order as defined with add_columns() or as keywords using kwdata.
        Don't mix these two approaches or you will get undefined behavior.
        :return:
        """
        pass
    
    @abstractmethod
    def store(self) -> None:
        """
        Closes the accumulated record, adds it to self.records and logs it to the logger
        """
        pass
    
    @abstractmethod
    def is_setup(self) -> bool:
        """
        :return: true if set_schema() has been called
        """
        pass
    
    @abstractmethod
    def get_column(self, name) -> []:
        """
        gets a column from the recorded data
        :param name: column name
        :return: iterable column
        """
        pass
    
    @abstractmethod
    def get_record(self, index) -> []:
        """
        :param index:
        :return: record at given index in the recorded data.
        """
        pass
    
    @abstractmethod
    def get_records(self) -> []:
        """
        :return: all records
        """
        pass
    
    @abstractmethod
    def get_column_map(self) -> {}:
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Must be called to dispose of an instance
        """
        pass
