import datetime
import time
from typing import List, Union


class TimeMonitor:
    def __init__(self):
        self.start: float = time.time()
        self.records: List[float] = []

    def begin(self):
        self.start = time.time()
        return self

    def step(self):
        self.records.append(time.time() - self.start)
        return self

    @property
    def latest(self):
        return self.records[-1]

    @property
    def string(self) -> Union[str, None]:
        if self.start is None or len(self.records) == 0:
            return None

        return str(datetime.timedelta(seconds=int(round(self.records[-1]))))

    @property
    def history(self) -> Union[List[str], None]:
        return [str(datetime.timedelta(seconds=int(round(record)))) for record in self.records]
