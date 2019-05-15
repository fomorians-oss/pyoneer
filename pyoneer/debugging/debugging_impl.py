import time


class Stopwatch:
    """
    Embarassingly simple stopwatch for measuring how long operations take. Great for fast and easy profiling.

    Example:

    with Stopwatch() as watch:
        time.sleep(1.0)
    print(stopwatch.duration) # ~1s
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
