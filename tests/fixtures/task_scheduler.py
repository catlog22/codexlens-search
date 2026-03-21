"""Task scheduling and execution framework."""
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any


class TaskPriority(Enum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1
    CRITICAL = 0


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class Task:
    priority: int
    name: str = field(compare=False)
    fn: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Any = field(default=None, compare=False)
    error: str = field(default="", compare=False)


class TaskScheduler:
    """Priority-based task scheduler with execution tracking."""

    def __init__(self):
        self._queue: list[Task] = []
        self._completed: list[Task] = []
        self._failed: list[Task] = []

    def add_task(
        self,
        name: str,
        fn: Callable,
        priority: TaskPriority = TaskPriority.MEDIUM,
        *args: Any,
        **kwargs: Any,
    ) -> Task:
        task = Task(
            priority=priority.value,
            name=name,
            fn=fn,
            args=args,
            kwargs=kwargs,
        )
        heapq.heappush(self._queue, task)
        return task

    def run_next(self) -> Task | None:
        if not self._queue:
            return None
        task = heapq.heappop(self._queue)
        task.status = TaskStatus.RUNNING
        try:
            task.result = task.fn(*task.args, **task.kwargs)
            task.status = TaskStatus.COMPLETED
            self._completed.append(task)
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._failed.append(task)
        return task

    def run_all(self) -> list[Task]:
        results = []
        while self._queue:
            task = self.run_next()
            if task:
                results.append(task)
        return results

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def completed_tasks(self) -> list[Task]:
        return list(self._completed)

    @property
    def failed_tasks(self) -> list[Task]:
        return list(self._failed)
