class WrongResultFormatException(Exception):
    def __init__(self, expected, got):
        message = f"Expected {expected}, but got {got}"
        super().__init__(message)


class EmptyResultSetException(Exception):
    def __init__(self, query):
        message = f"Empty result set for query: {query}"
        super().__init__(message)