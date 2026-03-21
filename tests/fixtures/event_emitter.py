"""Event-driven publish/subscribe system."""
from collections import defaultdict
from typing import Callable, Any


class EventEmitter:
    """Simple event emitter with listener registration and emission."""

    def __init__(self):
        self._listeners: dict[str, list[Callable]] = defaultdict(list)
        self._once_listeners: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, callback: Callable) -> None:
        self._listeners[event].append(callback)

    def once(self, event: str, callback: Callable) -> None:
        self._once_listeners[event].append(callback)

    def off(self, event: str, callback: Callable) -> bool:
        try:
            self._listeners[event].remove(callback)
            return True
        except ValueError:
            return False

    def emit(self, event: str, *args: Any, **kwargs: Any) -> int:
        count = 0
        for cb in self._listeners.get(event, []):
            cb(*args, **kwargs)
            count += 1
        for cb in self._once_listeners.pop(event, []):
            cb(*args, **kwargs)
            count += 1
        return count

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(event, [])) + len(
            self._once_listeners.get(event, [])
        )

    def clear(self, event: str | None = None) -> None:
        if event:
            self._listeners.pop(event, None)
            self._once_listeners.pop(event, None)
        else:
            self._listeners.clear()
            self._once_listeners.clear()
