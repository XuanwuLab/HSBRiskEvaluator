"""
Advanced display manager for organized output with concurrent support.
"""

import threading
import time
import sys
import os
from collections import deque
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging


class DisplayManager:
    """
    Manages organized display output with 3 sections:
    - Top: Rotating log messages (last N lines)
    - Middle: Current step information (continuously refreshed)
    - Bottom: tqdm progress bars (handled by tqdm itself)
    """

    def __init__(self, max_log_lines: int = 10, refresh_interval: float = 0.1):
        self.max_log_lines = max_log_lines
        self.refresh_interval = refresh_interval

        # Thread-safe storage
        self.lock = threading.RLock()
        self.log_buffer = deque(maxlen=max_log_lines)
        self.current_steps: Dict[str, str] = {}  # thread_id -> step_info
        self.active = False
        self.display_thread: Optional[threading.Thread] = None

        # Terminal info
        self.terminal_height = self._get_terminal_height()
        self.log_section_height = min(max_log_lines + 2, self.terminal_height // 3)
        self.step_section_height = max(3, self.terminal_height // 3)

        # Setup custom log handler
        self.log_handler = self._create_log_handler()

    def _get_terminal_height(self) -> int:
        """Get terminal height, fallback to 24 if can't determine"""
        try:
            return os.get_terminal_size().lines
        except OSError:
            return 24

    def _create_log_handler(self) -> logging.Handler:
        """Create a custom log handler that feeds into our display"""

        class DisplayLogHandler(logging.Handler):
            def __init__(self, display_manager):
                super().__init__()
                self.display_manager = display_manager

            def emit(self, record):
                try:
                    timestamp = datetime.fromtimestamp(record.created).strftime(
                        "%H:%M:%S"
                    )
                    level = record.levelname.ljust(7)
                    name = record.name.split(".")[-1][:15].ljust(15)
                    message = self.format(record)

                    log_line = f"[{timestamp}] {level} {name} | {message}"
                    self.display_manager.add_log_line(log_line)
                except Exception:
                    pass  # Avoid recursive errors in logging

        handler = DisplayLogHandler(self)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        return handler

    def start(self):
        """Start the display manager"""
        with self.lock:
            if self.active:
                return

            self.active = True

            # Add our log handler to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(self.log_handler)

            # Start display thread
            self.display_thread = threading.Thread(
                target=self._display_loop, daemon=True
            )
            self.display_thread.start()

    def stop(self):
        """Stop the display manager"""
        with self.lock:
            if not self.active:
                return

            self.active = False

            # Remove our log handler
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)

            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=1.0)

        # Clear screen sections
        self._clear_display()

    def add_log_line(self, line: str):
        """Add a log line to the rotating display"""
        with self.lock:
            # Truncate long lines
            if len(line) > 120:
                line = line[:117] + "..."
            self.log_buffer.append(line)

    def update_step(self, step_info: str, thread_id: Optional[str] = None):
        """Update step information for current thread"""
        if thread_id is None:
            thread_id = str(threading.get_ident())

        with self.lock:
            self.current_steps[thread_id] = step_info

    def clear_step(self, thread_id: Optional[str] = None):
        """Clear step information for current thread"""
        if thread_id is None:
            thread_id = str(threading.get_ident())

        with self.lock:
            self.current_steps.pop(thread_id, None)

    def _display_loop(self):
        """Main display refresh loop"""
        while self.active:
            try:
                self._refresh_display()
                time.sleep(self.refresh_interval)
            except Exception:
                pass  # Continue running even if display fails

    def _refresh_display(self):
        """Refresh the display with current information"""
        if not self.active:
            return

        with self.lock:
            # Save cursor position and clear the area we control
            print("\033[s", end="", flush=True)  # Save cursor position
            
            # Calculate terminal dimensions
            try:
                terminal_width = os.get_terminal_size().columns
            except OSError:
                terminal_width = 80
                
            total_display_lines = self.log_section_height + self.step_section_height
            
            # Move to top of terminal and clear our display area
            print("\033[H", end="", flush=True)  # Move to home position
            for i in range(total_display_lines):
                print("\033[K", end="", flush=True)  # Clear line
                if i < total_display_lines - 1:
                    print()  # Move to next line

            # Move back to top and start drawing
            print("\033[H", end="", flush=True)  # Move to home position
            
            # Draw log section
            separator = "=" * min(terminal_width - 1, 80)
            log_separator = "-" * min(terminal_width - 1, 80)
            
            print(separator)
            print("LOGS (Recent)")
            print(log_separator)

            # Show recent logs
            log_lines_to_show = self.log_section_height - 4  # Account for headers and separator
            recent_logs = list(self.log_buffer)[-log_lines_to_show:] if self.log_buffer else []
            
            for log_line in recent_logs:
                # Truncate to terminal width
                display_line = log_line[:terminal_width - 1] if len(log_line) > terminal_width - 1 else log_line
                print(display_line)
                
            # Fill remaining log section lines
            for _ in range(max(0, log_lines_to_show - len(recent_logs))):
                print()

            # Draw step section
            print(log_separator)
            print("CURRENT STEPS")
            print(log_separator)

            # Show active steps
            step_lines_to_show = self.step_section_height - 4  # Account for headers and separator
            active_steps = list(self.current_steps.items())[:step_lines_to_show]
            
            for thread_id, step_info in active_steps:
                thread_short = thread_id[-6:] if len(thread_id) > 6 else thread_id
                step_line = f"[{thread_short}] {step_info}"
                # Truncate to terminal width
                display_line = step_line[:terminal_width - 1] if len(step_line) > terminal_width - 1 else step_line
                print(display_line)
                
            # Fill remaining step section lines  
            for _ in range(max(0, step_lines_to_show - len(active_steps))):
                print()
                
            print(separator)
            
            # Move cursor to bottom of our display area for tqdm
            print(f"\033[{total_display_lines + 1};1H", end="", flush=True)
            sys.stdout.flush()

    def _clear_display(self):
        """Clear our display sections"""
        for i in range(self.log_section_height + self.step_section_height):
            print(f"\033[{i+1};1H\033[K", end="")


# Global display manager instance
_display_manager: Optional[DisplayManager] = None
_display_lock = threading.Lock()


def get_display_manager() -> DisplayManager:
    """Get or create the global display manager"""
    global _display_manager

    with _display_lock:
        if _display_manager is None:
            _display_manager = DisplayManager()

        return _display_manager


def start_organized_display():
    """Start organized display mode"""
    manager = get_display_manager()
    manager.start()


def stop_organized_display():
    """Stop organized display mode"""
    global _display_manager

    with _display_lock:
        if _display_manager is not None:
            _display_manager.stop()


def update_current_step(step_info: str, thread_id: Optional[str] = None):
    """Update current step for this thread"""
    manager = get_display_manager()
    manager.update_step(step_info, thread_id)


def clear_current_step(thread_id: Optional[str] = None):
    """Clear current step for this thread"""
    manager = get_display_manager()
    manager.clear_step(thread_id)


class StepContext:
    """Context manager for step tracking"""

    def __init__(self, step_info: str, thread_id: Optional[str] = None):
        self.step_info = step_info
        self.thread_id = thread_id or str(threading.get_ident())

    def __enter__(self):
        update_current_step(self.step_info, self.thread_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_current_step(self.thread_id)


# Enhanced tqdm class that works with organized display
try:
    from tqdm import tqdm as _tqdm

    class OrganizedTqdm(_tqdm):
        """tqdm that positions itself at the bottom and works with DisplayManager"""

        def __init__(self, *args, **kwargs):
            # Get the display manager to calculate proper positioning
            display_manager = get_display_manager()
            
            # Calculate position based on display manager's reserved space
            if display_manager.active:
                reserved_lines = display_manager.log_section_height + display_manager.step_section_height + 1
                kwargs.setdefault("position", reserved_lines)
            else:
                kwargs.setdefault("position", None)
                
            kwargs.setdefault("leave", False)  # Don't leave progress bars after completion
            kwargs.setdefault("dynamic_ncols", True)  # Adjust to terminal width
            kwargs.setdefault("miniters", 1)  # Update more frequently
            kwargs.setdefault(
                "bar_format",
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

            super().__init__(*args, **kwargs)

        @classmethod  
        def write(cls, s, file=None, end="\n", nolock=False):
            """Override write to position output correctly"""
            # Write below our organized sections
            display_manager = get_display_manager()
            if display_manager.active:
                reserved_lines = display_manager.log_section_height + display_manager.step_section_height + 2
                print(f"\033[{reserved_lines};1H{s}", end=end, file=file or sys.stdout, flush=True)
            else:
                print(s, end=end, file=file or sys.stdout, flush=True)

    # Replace the tqdm class for our organized version
    organized_tqdm = OrganizedTqdm
    TQDM_AVAILABLE = True

except ImportError:
    # Fallback when tqdm is not available
    def organized_tqdm(iterable, *args, **kwargs):
        return iterable

    TQDM_AVAILABLE = False
