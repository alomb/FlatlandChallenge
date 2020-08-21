from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def step(self, **args):
        pass

    @abstractmethod
    def act(self, state, **args):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass