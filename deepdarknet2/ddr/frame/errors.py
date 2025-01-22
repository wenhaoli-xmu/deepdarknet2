class NoCalcPath(Exception):
    def __str__(self):
        return f'没有相应的计算路径'


class YamlParseError(Exception):
    ...


class StopTraining(Exception):
    ...


class HessianNegtiveDefinite(Exception):
    ...
