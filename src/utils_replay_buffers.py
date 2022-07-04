
import random
import numpy as np
from collections import deque

##################################################################################################
#.Replay Buffers
##################################################################################################

#exp = (state, next_state, action, reward)
class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self.repeat_count = 0

    def add(self, experience):
        #if experience in self.buffer:
        #equal = self._check_equal(experience)
        #if(equal):
        #    self.repeat_count+=1
        #    return 0
            
        self.buffer.append(experience)
        return 1

    #TODO Fix
    def _check_equal(self,experience):
        for tup in self.buffer:
            check = False
            for i,element in enumerate(tup):
                if(element==experience[i]):
                    check = True
                else:
                    check = False
                    break

            if(check):
                return True
        
        return False

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

#Samples a uniform number of samples based on actions taken
class BalancedMemory(object):
    def __init__(self, memory_size,num_actions):
        self.memory_size = int(memory_size/num_actions)
        self.buffer_size = num_actions
        self.buffer = [deque(maxlen=self.memory_size) for _ in range(num_actions)]


    def add(self, experience) :
        action = experience[2]
        self.buffer[action].append(experience)

    #Warning: return min(len) * num_actions to guarantee uniform values on the batch
    def size(self):

        minimum_size= min([len(deq_buff) for deq_buff in self.buffer])

        return minimum_size*self.buffer_size

    def full_size(self):
        total_size = 0
        for deq_buff in self.buffer:
            total_size += len(deq_buff)

        return total_size

    def sample(self, batch_size):
        
        batch_sample = []
        buffer_sample_size = int((batch_size/self.buffer_size))
        leftover =  (batch_size % self.buffer_size)

        for i in range(self.buffer_size):
            indexes = np.random.choice(np.arange(len(self.buffer[i])), size=buffer_sample_size, replace=False)
            batch_sample+= [self.buffer[i][idx] for idx in indexes]
        
        random.shuffle(batch_sample) #shuffling batch

        return batch_sample

    def clear(self):
        for i in range(self.buffer_size):
            self.buffer[i].clear()

#exp = (state, next_state, action, reward) + "priority", "probability", "weight","index"
class PrioritizedMemory(): 

    def __init__(self, memory_size):

        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.used_size = 0
        self.e = 0.01 # ensuring that the sample has some non-zero probability of being drawn
        self.alpha = 0.6 #prioritization level
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def size(self):
        return self.used_size
    
    def add(self, error, experience):
        p = self._get_priority(error)
        self.tree.add(p, experience)

        self.used_size+=1

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            while True: #!optimize later
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if(not isinstance(data,int)):
                    break
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


##################################################################################################
#.Structure
##################################################################################################
# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])