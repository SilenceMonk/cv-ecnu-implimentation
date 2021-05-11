import abc  # 利用abc模块实现抽象类
from abc import ABC


class BaseMeasure(metaclass=abc.ABCMeta):

    @abc.abstractmethod  # 定义抽象方法，无需实现功能
    def get_index_mismatch(self):
        pass

    @abc.abstractmethod  # 定义抽象方法，无需实现功能
    def get_error_rate(self):
        pass


class MeasureKnn(BaseMeasure):
    def get_index_mismatch(self):
        pass

    def get_error_rate(self):
        pass


class MeasurePerceptron(BaseMeasure):
    def get_index_mismatch(self):
        pass

    def get_error_rate(self):
        pass
