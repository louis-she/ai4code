from ignite.base.mixins import Serializable


class SerializableDict(Serializable):

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state

    def __getitem__(self, key):
        return self._state[key]
