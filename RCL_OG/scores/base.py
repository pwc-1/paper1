import sys

from collections import namedtuple
from abc import ABC, abstractmethod

LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])

class BaseScore(ABC):
    def __init__(self, data):
        self.data = data
        self.num_variables = data.shape[1]

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    @abstractmethod
    def get_local_scores(self, target, indices, indices_after=None):
        pass


class BasePrior(ABC):
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value
