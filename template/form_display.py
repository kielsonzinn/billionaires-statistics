from abc import ABC, abstractmethod


class FormDisplay(ABC):
    _form = None
    _data = None

    def show(self):
        self.create_pf()
        self._form.show()

    def set_data(self, data):
        self._data = data

    @abstractmethod
    def create_pf(self):
        pass
