import math
import numpy as np


class replaybuffer:
    def __init__(self, num_variables, capacity):
        self.capacity = capacity
        self.num_variables = num_variables
        nbytes = math.ceil(num_variables / 8)
        dtype = np.dtype([
            ('state', np.uint8, (nbytes,)),
            ('action', np.uint8, (nbytes,)),
            ('reward', np.float_, (1,)),
            ('next_state', np.uint8, (nbytes,)),
        ])
        self._reply = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
    # add states, actions, rewards and next_states to buffer
    def add(self, states, actions, rewards, next_states):
        num_samples = states.shape[0]
        add_idx = np.arange(self._index, self._index+num_samples) % self.capacity
        self._index = (self._index+num_samples) % self.capacity
        self._reply['state'][add_idx] = self.encode(states)
        self._reply['action'][add_idx] = self.encode(actions)
        self._reply['reward'][add_idx] = rewards
        self._reply['next_state'][add_idx] = self.encode(next_states)
    # sample states, actions, rewards and next_states from buffer
    def sample(self, batch_size):
        index = np.random.choice(self._index, batch_size, replace=False)
        sample = self._reply[index]
        return {
            'state': self.decode(sample['state']),
            'action': self.decode(sample['action']),
            'reward': sample['reward'],
            'next_state': self.decode(sample['next_state']),
        }

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables)
        return np.packbits(encoded.astype(int), axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables)
        decoded = decoded.reshape(*encoded.shape[:-1], 1, self.num_variables)
        return decoded.astype(dtype)

    def length(self):
        return self._index+1


