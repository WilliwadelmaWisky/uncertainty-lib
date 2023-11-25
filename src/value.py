
class Value:
    def __init__(self, val, err):
        self._val = val
        self._err = err

    def get_error(self) -> float:
        return self._err

    def get_value(self):
        return self._val

    def get_min_value(self) -> float:
        return self.get_value() - self.get_error()

    def get_max_value(self) -> float:
        return self.get_value() + self.get_error()
