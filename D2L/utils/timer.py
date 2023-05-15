import time
import numpy as np


class Timer:
    """Record multiple run times"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start timer"""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the total time"""
        return sum(self.times)

    def cumsum(self):
        """Return cumulative time"""
        return np.array(self.times).cumsum().tolist()