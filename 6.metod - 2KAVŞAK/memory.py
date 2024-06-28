import random

class Memory:
    def __init__(self, size_max, size_min):
        self._samples_1 = []
        self._samples_2 = []
        self._size_max = size_max
        self._size_min = size_min


    def add_sample_1(self, sample):
        """
        Add a sample into the memory
        """
        self._samples_1.append(sample)
        if self._size_now_1() > self._size_max:
            self._samples_1.pop(0)  # if the length is greater than the size of memory, remove the oldest element
            
            
    def add_sample_2(self, sample):
        """
        Add a sample into the memory
        """
        self._samples_2.append(sample)
        if self._size_now_2() > self._size_max:
            self._samples_2.pop(0)  # if the length is greater than the size of memory, remove the oldest element


    def get_samples_1(self, n):
        """
        Get n samples randomly from the memory
        """
        if self._size_now_1() < self._size_min:
            return []

        if n > self._size_now_1():
            return random.sample(self._samples_1, self._size_now_1())  # get all the samples
        else:
            return random.sample(self._samples_1, n)  # get "batch size" number of samples
        
        
    def get_samples_2(self, n):
        """
        Get n samples randomly from the memory
        """
        if self._size_now_2() < self._size_min:
            return []

        if n > self._size_now_2():
            return random.sample(self._samples_2, self._size_now_2())  # get all the samples
        else:
            return random.sample(self._samples_2, n)  # get "batch size" number of samples


    def _size_now_1(self):
        """
        Check how full the memory is
        """
        return len(self._samples_1)
    
    
    def _size_now_2(self):
        """
        Check how full the memory is
        """
        return len(self._samples_2)