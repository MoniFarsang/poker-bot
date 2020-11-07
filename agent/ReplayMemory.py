from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory(object):
    ''' 
    Replay memory for saving transitions
    '''
    def __init__(self, capacity, batch_size):
        ''' 
        Initialize ReplayMemory

        :param int capacity: the size of the memory buffer
        :param int batch_size: the size of the batches
        '''
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''
        Save a transition into memory
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Choose random sample from the memory with size of the batch size
        '''
        samples = random.sample(self.memory, batch_size)
        return map(np.array, zip(*samples))

