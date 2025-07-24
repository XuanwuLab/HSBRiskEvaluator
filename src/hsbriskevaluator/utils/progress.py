"""
Progress tracking utilities for HSBRiskEvaluator.

This module provides a two-layer progress tracking system:
1. Todo tracking with checkmark display (✅ todo1 ✅ todo2 [*] todo3)
2. Progress bars using tqdm for detailed progress within each task
"""

import sys
import time
from typing import List, Dict, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from contextlib import contextmanager

# Try to import tqdm, fall back gracefully if not available
try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as atqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create mock tqdm classes for when tqdm is not available
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.total = total
            self.n = 0
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self, n=1):
            self.n += n
            
        def set_description(self, desc):
            self.desc = desc
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update(1)
    
    class atqdm(tqdm):
        pass


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that works with tqdm progress bars.
    
    This handler ensures that log messages don't interfere with progress bars
    by using tqdm.write() when tqdm is available.
    """
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if TQDM_AVAILABLE:
                # Use tqdm.write to avoid interfering with progress bars
                tqdm.write(msg)
            else:
                # Fall back to regular print
                print(msg)
        except Exception:
            self.handleError(record)


class ProgressLoggingContext:
    """
    Context manager that temporarily redirects logging to be compatible with progress bars.
    """
    
    _lock = threading.Lock()
    _original_handlers = None
    _tqdm_handler = None
    _context_count = 0
    
    def __init__(self):
        self.entered = False
    
    def __enter__(self):
        with self._lock:
            self.__class__._context_count += 1
            if self.__class__._context_count == 1:
                # First context entry - setup tqdm logging
                self._setup_tqdm_logging()
            self.entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.entered:
            with self._lock:
                self.__class__._context_count -= 1
                if self.__class__._context_count == 0:
                    # Last context exit - restore original logging
                    self._restore_original_logging()
                self.entered = False
    
    @classmethod
    def _setup_tqdm_logging(cls):
        """Setup tqdm-compatible logging for all loggers"""
        if not TQDM_AVAILABLE:
            return  # No need to change logging if tqdm is not available
            
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Store original handlers
        cls._original_handlers = root_logger.handlers.copy()
        
        # Create and add tqdm handler
        cls._tqdm_handler = TqdmLoggingHandler()
        cls._tqdm_handler.setLevel(logging.INFO)
        
        # Replace all handlers with our tqdm handler
        root_logger.handlers.clear()
        root_logger.addHandler(cls._tqdm_handler)
        
        # Also handle specific loggers that might be used in the collector
        for logger_name in ['hsbriskevaluator', 'github', 'urllib3']:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.addHandler(cls._tqdm_handler)
            logger.propagate = True
    
    @classmethod
    def _restore_original_logging(cls):
        """Restore original logging configuration"""
        if cls._original_handlers is not None:
            # Get the root logger
            root_logger = logging.getLogger()
            
            # Remove our handler
            if cls._tqdm_handler in root_logger.handlers:
                root_logger.removeHandler(cls._tqdm_handler)
            
            # Restore original handlers
            root_logger.handlers.clear()
            root_logger.handlers.extend(cls._original_handlers)
            
            # Restore specific loggers
            for logger_name in ['hsbriskevaluator', 'github', 'urllib3']:
                logger = logging.getLogger(logger_name)
                logger.handlers.clear()
                logger.propagate = True
            
            # Clean up
            cls._original_handlers = None
            cls._tqdm_handler = None


@contextmanager
def progress_logging():
    """Context manager for tqdm-compatible logging"""
    with ProgressLoggingContext():
        yield

logger = logging.getLogger(__name__)


class TodoStatus(Enum):
    """Status of a todo item"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """A single todo item with status tracking"""
    id: str
    description: str
    status: TodoStatus = TodoStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the task if completed"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
        
    @property
    def status_symbol(self) -> str:
        """Get the status symbol for display"""
        symbols = {
            TodoStatus.PENDING: "⏸️",
            TodoStatus.IN_PROGRESS: "🔄",
            TodoStatus.COMPLETED: "✅",
            TodoStatus.FAILED: "❌",
            TodoStatus.SKIPPED: "⏭️"
        }
        return symbols.get(self.status, "❓")
        
    def start(self):
        """Mark the todo as started"""
        self.status = TodoStatus.IN_PROGRESS
        self.start_time = time.time()
        
    def complete(self):
        """Mark the todo as completed"""
        self.status = TodoStatus.COMPLETED
        self.end_time = time.time()
        
    def fail(self, error: Optional[str] = None):
        """Mark the todo as failed"""
        self.status = TodoStatus.FAILED
        self.end_time = time.time()
        self.error = error
        
    def skip(self):
        """Mark the todo as skipped"""
        self.status = TodoStatus.SKIPPED
        self.end_time = time.time()


class ProgressTracker:
    """
    A progress tracker that manages todo items and progress bars
    """
    
    def __init__(self, name: str = "Progress", show_progress: bool = True):
        self.name = name
        self.show_progress = show_progress
        self.todos: Dict[str, TodoItem] = {}
        self.todo_order: List[str] = []
        self._current_todo: Optional[str] = None
        self._progress_bar: Optional[tqdm] = None
        self._logging_context: Optional[ProgressLoggingContext] = None
        
        # Setup logging context if progress is enabled
        if self.show_progress:
            self._logging_context = ProgressLoggingContext()
            self._logging_context.__enter__()
        
    def add_todo(self, todo_id: str, description: str) -> TodoItem:
        """Add a new todo item"""
        todo = TodoItem(id=todo_id, description=description)
        self.todos[todo_id] = todo
        self.todo_order.append(todo_id)
        self._update_display()
        return todo
        
    def add_todos(self, todos: List[tuple[str, str]]) -> List[TodoItem]:
        """Add multiple todo items at once"""
        todo_items = []
        for todo_id, description in todos:
            todo_items.append(self.add_todo(todo_id, description))
        return todo_items
        
    def start_todo(self, todo_id: str):
        """Start working on a specific todo"""
        if todo_id not in self.todos:
            raise ValueError(f"Todo {todo_id} not found")
            
        # Complete previous todo if it was in progress
        if self._current_todo and self._current_todo in self.todos:
            current = self.todos[self._current_todo]
            if current.status == TodoStatus.IN_PROGRESS:
                current.complete()
        
        self._current_todo = todo_id
        self.todos[todo_id].start()
        self._update_display()
        
    def complete_todo(self, todo_id: str):
        """Mark a todo as completed"""
        if todo_id not in self.todos:
            raise ValueError(f"Todo {todo_id} not found")
            
        self.todos[todo_id].complete()
        if self._current_todo == todo_id:
            self._current_todo = None
        self._update_display()
        
    def fail_todo(self, todo_id: str, error: Optional[str] = None):
        """Mark a todo as failed"""
        if todo_id not in self.todos:
            raise ValueError(f"Todo {todo_id} not found")
            
        self.todos[todo_id].fail(error)
        if self._current_todo == todo_id:
            self._current_todo = None
        self._update_display()
        
    def skip_todo(self, todo_id: str):
        """Mark a todo as skipped"""
        if todo_id not in self.todos:
            raise ValueError(f"Todo {todo_id} not found")
            
        self.todos[todo_id].skip()
        if self._current_todo == todo_id:
            self._current_todo = None
        self._update_display()
        
    def create_progress_bar(self, total: int, desc: str = "", **kwargs) -> tqdm:
        """Create a progress bar for the current operation"""
        if not self.show_progress:
            return tqdm(total=total, desc=desc, disable=True, **kwargs)
        
        return tqdm(total=total, desc=desc, leave=False, **kwargs)
        
    def create_async_progress_bar(self, total: int, desc: str = "", **kwargs) -> atqdm:
        """Create an async progress bar for the current operation"""
        if not self.show_progress:
            return atqdm(total=total, desc=desc, disable=True, **kwargs)
        
        return atqdm(total=total, desc=desc, leave=False, **kwargs)
        
    def cleanup(self):
        """Clean up logging context"""
        if self._logging_context:
            try:
                self._logging_context.__exit__(None, None, None)
                self._logging_context = None
            except Exception:
                pass  # Ignore cleanup errors
    
    def _update_display(self):
        """Update the todo display"""
        if not self.show_progress:
            return
            
        # Create the todo status line
        todo_parts = []
        for todo_id in self.todo_order:
            todo = self.todos[todo_id]
            symbol = todo.status_symbol
            desc = todo.description[:20] + "..." if len(todo.description) > 20 else todo.description
            todo_parts.append(f"{symbol} {desc}")
        
        todo_line = " ".join(todo_parts)
        
        # Use tqdm.write if available and inside logging context, otherwise print normally
        if TQDM_AVAILABLE and self._logging_context:
            tqdm.write(f"{self.name}: {todo_line}")
        else:
            # Print the status line (overwrite previous line if possible)
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                print(f"\r{self.name}: {todo_line}", end="", flush=True)
            else:
                print(f"{self.name}: {todo_line}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all todos"""
        summary = {
            "total": len(self.todos),
            "completed": sum(1 for t in self.todos.values() if t.status == TodoStatus.COMPLETED),
            "failed": sum(1 for t in self.todos.values() if t.status == TodoStatus.FAILED),
            "skipped": sum(1 for t in self.todos.values() if t.status == TodoStatus.SKIPPED),
            "in_progress": sum(1 for t in self.todos.values() if t.status == TodoStatus.IN_PROGRESS),
            "pending": sum(1 for t in self.todos.values() if t.status == TodoStatus.PENDING),
            "total_duration": sum(t.duration for t in self.todos.values() if t.duration),
            "todos": self.todos
        }
        return summary
        
    def print_summary(self):
        """Print a final summary"""
        summary = self.get_summary()
        
        # Choose the appropriate print function
        if TQDM_AVAILABLE and self._logging_context:
            write_func = tqdm.write
        else:
            write_func = print
        
        write_func(f"\n{self.name} Summary:")
        write_func(f"  Total: {summary['total']}")
        write_func(f"  ✅ Completed: {summary['completed']}")
        if summary['failed'] > 0:
            write_func(f"  ❌ Failed: {summary['failed']}")
        if summary['skipped'] > 0:
            write_func(f"  ⏭️ Skipped: {summary['skipped']}")
        if summary['total_duration']:
            write_func(f"  ⏱️ Total Duration: {summary['total_duration']:.2f}s")
            
        # Print failed todos with errors
        failed_todos = [t for t in self.todos.values() if t.status == TodoStatus.FAILED]
        if failed_todos:
            write_func("\nFailed Tasks:")
            for todo in failed_todos:
                write_func(f"  ❌ {todo.description}")
                if todo.error:
                    write_func(f"     Error: {todo.error}")
        
        # Clean up logging context after printing summary
        self.cleanup()


class ProgressContext:
    """
    Context manager for progress tracking
    """
    
    def __init__(self, tracker: ProgressTracker, todo_id: str):
        self.tracker = tracker
        self.todo_id = todo_id
        
    def __enter__(self):
        self.tracker.start_todo(self.todo_id)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = str(exc_val) if exc_val else "Unknown error"
            self.tracker.fail_todo(self.todo_id, error_msg)
        else:
            self.tracker.complete_todo(self.todo_id)
        return False  # Don't suppress exceptions


def create_progress_tracker(name: str = "Progress", show_progress: bool = True) -> ProgressTracker:
    """Create a new progress tracker with automatic logging management"""
    return ProgressTracker(name=name, show_progress=show_progress)


# Utility functions for common progress tracking patterns
async def track_async_tasks(
    tasks: List[Callable[[], Awaitable[Any]]], 
    descriptions: List[str],
    tracker: Optional[ProgressTracker] = None,
    name: str = "Async Tasks"
) -> List[Any]:
    """
    Track multiple async tasks with progress display
    
    Args:
        tasks: List of async functions to execute
        descriptions: Description for each task
        tracker: Existing progress tracker or None to create new one
        name: Name for the tracker
        
    Returns:
        List of results from each task
    """
    if tracker is None:
        tracker = create_progress_tracker(name)
        
    # Add todos for each task
    todo_ids = []
    for i, desc in enumerate(descriptions):
        todo_id = f"task_{i}"
        tracker.add_todo(todo_id, desc)
        todo_ids.append(todo_id)
    
    results = []
    for i, task in enumerate(tasks):
        todo_id = todo_ids[i]
        try:
            with ProgressContext(tracker, todo_id):
                result = await task()
                results.append(result)
        except Exception as e:
            logger.error(f"Task {descriptions[i]} failed: {e}")
            results.append(None)
    
    return results