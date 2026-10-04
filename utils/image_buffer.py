"""Image-specific ring buffer backed by a pre-allocated numpy array."""
import threading
import numpy as np
from typing import Optional, Tuple

class ImageRingBuffer:
    """Fixed-capacity ring buffer for satellite image tensors. O(1) push/pop."""
    def __init__(self, capacity=50, shape=(3,64,64)):
        self.capacity = capacity
        self._buf    = np.zeros((capacity,*shape), dtype=np.float32)
        self._labels = np.full(capacity, -1, dtype=np.int32)
        self._head   = 0; self._tail = 0; self._size = 0
        self._lock   = threading.Lock()
        self._pushed = 0; self._overflows = 0

    def push(self, img: np.ndarray, label: int = -1):
        with self._lock:
            if self._size == self.capacity:
                self._tail = (self._tail+1) % self.capacity
                self._size -= 1; self._overflows += 1
            self._buf[self._head]    = img
            self._labels[self._head] = label
            self._head = (self._head+1) % self.capacity
            self._size += 1; self._pushed += 1

    def pop(self) -> Optional[Tuple[np.ndarray,int]]:
        with self._lock:
            if self._size == 0: return None
            img   = self._buf[self._tail].copy()
            label = int(self._labels[self._tail])
            self._tail = (self._tail+1) % self.capacity
            self._size -= 1
            return img, label

    @property
    def size(self):
        with self._lock: return self._size
    @property
    def fill_ratio(self):
        with self._lock: return self._size / self.capacity

    def stats(self):
        with self._lock:
            return {"capacity":self.capacity,"size":self._size,
                    "fill":round(self._size/self.capacity,3),
                    "pushed":self._pushed,"overflows":self._overflows,
                    "head":self._head,"tail":self._tail}
