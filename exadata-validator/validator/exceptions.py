import sys
from contextlib import contextmanager
from enum import IntEnum
from functools import wraps


class DataBaseError(Exception):
    """Legacy DB error type."""

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class UserInputError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class FileContentError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class InvalidDatasetError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class InvalidDataModelError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class ForeignKeyError(Exception):
    """Legacy FK error type."""

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(message)


class ExitCode(IntEnum):
    OK = 0
    USER_ERROR = 64
    DB_ERROR = 65
    FILE_ERROR = 66


def handle_errors(func):
    @contextmanager
    def _handle_errors():
        try:
            yield
        except UserInputError as exc:
            print("User input error:\n")
            print(f"\t{exc.message}")
            sys.exit(ExitCode.USER_ERROR)
        except DataBaseError as exc:
            print("Database error:\n")
            print(f"\t{exc.message}")
            sys.exit(ExitCode.DB_ERROR)
        except (FileContentError, InvalidDatasetError, InvalidDataModelError) as exc:
            print(f"\nValidation error: {exc.message}")
            sys.exit(ExitCode.FILE_ERROR)
        except ForeignKeyError as exc:
            print("Foreign key error:\n")
            print(f"\t{exc.message}")
            sys.exit(ExitCode.USER_ERROR)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _handle_errors():
            return func(*args, **kwargs)

    return wrapper
