"""DSA utilities: Ring Buffer, Sliding Window + Monotonic Deque, Min-Heap Scheduler."""
import heapq, time, threading, uuid
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List
from enum import IntEnum


# ── Ring Buffer ───────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-size circular buffer. O(1) push/pop. No allocation after init."""
    def __init__(self, capacity=50):
        self.capacity = capacity
        self._buf  = [None] * capacity
        self._head = 0; self._tail = 0; self._size = 0
        self._lock = threading.Lock()
        self._pushed = 0; self._overflows = 0

    def push(self, item):
        with self._lock:
            if self._size == self.capacity:
                self._tail = (self._tail + 1) % self.capacity
                self._size -= 1; self._overflows += 1
            self._buf[self._head] = item
            self._head = (self._head + 1) % self.capacity
            self._size += 1; self._pushed += 1

    def pop(self):
        with self._lock:
            if self._size == 0: return None
            item = self._buf[self._tail]
            self._tail = (self._tail + 1) % self.capacity
            self._size -= 1
            return item

    def peek(self):
        with self._lock:
            return self._buf[self._tail] if self._size > 0 else None

    @property
    def size(self):
        with self._lock: return self._size

    @property
    def fill_ratio(self):
        with self._lock: return self._size / self.capacity

    def stats(self):
        with self._lock:
            return {"capacity": self.capacity, "size": self._size,
                    "fill": round(self._size/self.capacity, 3),
                    "pushed": self._pushed, "overflows": self._overflows,
                    "head": self._head, "tail": self._tail}


# ── Sliding Window + Monotonic Deque ─────────────────────────────────────────

class SlidingWindow:
    """O(1) rolling mean/RMS/max/min via running sums + monotonic deque."""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self._w    = deque()
        self._sum  = 0.0; self._ssq = 0.0
        self._maxd = deque()   # decreasing → front = max
        self._mind = deque()   # increasing → front = min

    def push(self, v: float):
        if len(self._w) == self.window_size:
            old = self._w[0]
            self._sum -= old; self._ssq -= old*old
            self._w.popleft()
            if self._maxd and self._maxd[0] == old: self._maxd.popleft()
            if self._mind and self._mind[0] == old: self._mind.popleft()
        self._w.append(v); self._sum += v; self._ssq += v*v
        while self._maxd and self._maxd[-1] < v: self._maxd.pop()
        self._maxd.append(v)
        while self._mind and self._mind[-1] > v: self._mind.pop()
        self._mind.append(v)

    @property
    def mean(self):    n=len(self._w); return self._sum/n if n else 0.0
    @property
    def rms(self):     n=len(self._w); return float(np.sqrt(self._ssq/n)) if n else 0.0
    @property
    def window_max(self): return self._maxd[0] if self._maxd else 0.0
    @property
    def window_min(self): return self._mind[0] if self._mind else 0.0
    @property
    def std(self):
        n=len(self._w)
        return float(np.sqrt(max(0, self._ssq/n - (self._sum/n)**2))) if n>1 else 0.0
    @property
    def size(self): return len(self._w)

    def stats(self):
        return {"mean": round(self.mean,4), "rms": round(self.rms,4),
                "max": round(self.window_max,4), "min": round(self.window_min,4),
                "std": round(self.std,4), "size": self.size}


# ── Min-Heap Priority Scheduler ───────────────────────────────────────────────

class Priority(IntEnum):
    CRITICAL   = 0
    HIGH       = 1
    NORMAL     = 2
    LOW        = 3
    BACKGROUND = 4

@dataclass(order=True)
class Task:
    priority:   int
    timestamp:  float = field(compare=True)
    task_id:    str   = field(compare=False)
    name:       str   = field(compare=False)
    fn:         Callable = field(compare=False, repr=False)
    args:       tuple = field(default_factory=tuple, compare=False, repr=False)
    result:     Any   = field(default=None, compare=False)
    error:      Optional[Exception] = field(default=None, compare=False)
    created_at: float = field(default_factory=time.perf_counter, compare=False)
    finished_at:Optional[float] = field(default=None, compare=False)

class PriorityScheduler:
    """Min-heap task scheduler. O(log n) enqueue/dequeue."""
    def __init__(self, name="sched"):
        self.name = name
        self._heap: List = []
        self._lock = threading.Lock()
        self._history: List[dict] = []
        self._stats = {"enqueued":0, "executed":0, "errors":0}

    def enqueue(self, fn, name="task", priority=Priority.NORMAL, args=()):
        task = Task(priority=int(priority), timestamp=time.perf_counter(),
                    task_id=str(uuid.uuid4())[:8], name=name, fn=fn, args=args)
        with self._lock:
            heapq.heappush(self._heap, (task.priority, task.timestamp, task))
            self._stats["enqueued"] += 1
        return task.task_id

    def run_next(self):
        with self._lock:
            if not self._heap: return None
            _, _, task = heapq.heappop(self._heap)
        try:
            task.result = task.fn(*task.args)
        except Exception as e:
            task.error = e
            with self._lock: self._stats["errors"] += 1
        finally:
            task.finished_at = time.perf_counter()
            ms = (task.finished_at - task.created_at)*1000
            with self._lock:
                self._stats["executed"] += 1
                self._history.append({"name":task.name,"priority":task.priority,
                                       "wait_ms":round(ms,3),"ok":task.error is None})
                if len(self._history)>300: self._history=self._history[-300:]
        return task

    def run_all(self):
        out = []
        while self.size > 0:
            t = self.run_next()
            if t: out.append(t)
        return out

    @property
    def size(self):
        with self._lock: return len(self._heap)
    @property
    def is_empty(self): return self.size == 0

    def get_history(self, n=20):
        with self._lock: return list(self._history[-n:])
    def stats(self):
        with self._lock: return {**self._stats, "pending": len(self._heap)}
