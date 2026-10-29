class WorkerUnresponsiveError(Exception):
    def __init__(self):
        message = (
            "One of the workers participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class WorkerTaskTimeoutError(Exception):
    def __init__(self):
        message = (
            f"One of the worker tasks in the algorithm execution took longer to finish than "
            f"the timeout. This could be caused by a high load or by an experiment with "
            f"too much data. Please try again or increase the timeout."
        )
        super().__init__(message)
        self.message = message


class GlobalWorkerUnavailableError(Exception):
    def __init__(self):
        message = (
            "The global worker is not part of the federation right now, so "
            "this analysis cannot run. Please try again in a few minutes."
        )
        super().__init__(message)
        self.message = message
