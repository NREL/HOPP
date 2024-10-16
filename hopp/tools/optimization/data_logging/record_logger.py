from abc import abstractmethod


class RecordLogger:
    """
    Logs data via a write() method. Should be closed when you are done.
    """
    
    def __del__(self) -> None:
        self.close()  # tries to close the logger if it isn't already closed
    
    @abstractmethod
    def write(self, data) -> None:
        """
        Records data into the logger.
        """
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """
        Flushes accumulated data to the log.
        """
        pass
    
    def close(self) -> None:
        """
        Close this logger. Loggers should be disposed when you are done with them.
        May be called more than once.
        """
        pass
    
    def write_and_flush(self, data) -> None:
        self.write(data)
        self.flush()
