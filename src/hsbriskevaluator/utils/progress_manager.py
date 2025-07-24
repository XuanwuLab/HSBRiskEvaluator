"""
Multi-process and async compatible progress and logging manager.
Completely separates logs from progress bars to prevent interference.
"""

import os
import sys
import logging
import threading
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing import current_process
from tqdm import tqdm
from contextlib import contextmanager
from datetime import datetime


class ProgressManager:
    """
    Centralized progress and logging manager that handles:
    1. Multi-process tqdm coordination 
    2. Async task progress tracking
    3. Complete logging/tqdm separation with file-based logging
    """
    
    _instance = None
    _lock = threading.Lock()
    _log_file = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.process_name = current_process().name
        self.process_id = os.getpid()
        self._progress_bars: Dict[str, tqdm] = {}
        self._main_bar: Optional[tqdm] = None
        self._position_counter = 0
        self._setup_logging()
        self._reserve_terminal_space()
    
    def _setup_logging(self):
        """Setup file-based logging to completely avoid terminal interference"""
        # Create a dedicated log file for this session
        if not ProgressManager._log_file:
            log_dir = Path(tempfile.gettempdir()) / "hsbriskevaluator_logs"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ProgressManager._log_file = log_dir / f"hsbriskevaluator_{timestamp}.log"
        
        # Configure logger to write only to file
        self.logger = logging.getLogger(f"HSBRisk.{self.process_name}")
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(ProgressManager._log_file)
        file_formatter = logging.Formatter(
            f'%(asctime)s - {self.process_name}[{self.process_id}] - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Print log file location once
        if self.process_name == "MainProcess":
            print(f"\n📝 Logs are being written to: {ProgressManager._log_file}")
            print("💡 You can monitor logs with: tail -f {}\n".format(ProgressManager._log_file))
    
    def _reserve_terminal_space(self):
        """Reserve terminal space for progress bars"""
        if self.process_name == "MainProcess":
            # Clear some space for progress bars
            print("\n" * 3)  # Reserve 3 lines for progress bars
    
    @contextmanager
    def create_main_progress(self, total: int, desc: str = "Processing", unit: str = "items"):
        """Create main progress bar for the current process with stable positioning"""
        if self.process_name != "MainProcess":
            # Subprocesses use file logging only, no progress bars
            yield self._create_file_only_counter(total, desc, unit)
            return
        
        # Main process gets a stable progress bar at the bottom
        self._main_bar = tqdm(
            total=total,
            desc=f"📊 {desc}",
            unit=unit,
            position=0,  # Always at position 0 (bottom)
            leave=True,
            dynamic_ncols=True,
            ncols=100,  # Fixed width to prevent resizing
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        try:
            yield self._main_bar
        finally:
            if self._main_bar:
                self._main_bar.close()
                self._main_bar = None
    
    @contextmanager
    def create_sub_progress(self, total: Optional[int], desc: str, unit: str = "items", 
                          parent_id: Optional[str] = None):
        """Create sub-progress bar with stable positioning"""
        if self.process_name != "MainProcess":
            # Subprocesses use file-only logging
            yield self._create_file_only_counter(total, desc, unit)
            return
        
        # Main process gets visual sub-progress bars with fixed positions
        self._position_counter += 1
        bar_id = f"{parent_id or 'sub'}_{self._position_counter}"
        
        # Create sub-progress bar above main progress bar
        progress_bar = tqdm(
            total=total,
            desc=f"  └─ {desc}",
            unit=unit,
            position=self._position_counter,  # Above main bar
            leave=False,
            dynamic_ncols=False,
            ncols=90,  # Slightly smaller than main bar
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]'
        )
        
        self._progress_bars[bar_id] = progress_bar
        
        try:
            yield progress_bar
        finally:
            if bar_id in self._progress_bars:
                self._progress_bars[bar_id].close()
                del self._progress_bars[bar_id]
    
    def _create_file_only_counter(self, total: Optional[int], desc: str, unit: str):
        """Create a counter that only logs to file without terminal output"""
        class FileOnlyCounter:
            def __init__(self, total, desc, unit, logger):
                self.n = 0
                self.total = total
                self.desc = desc
                self.unit = unit
                self.logger = logger
                self._last_log_percent = -1
                self.logger.info(f"Started: {desc} (target: {total or 'unknown'} {unit})")
                
            def update(self, n=1):
                self.n += n
                if self.total:
                    percent = int((self.n / self.total) * 100)
                    # Log every 10% progress
                    if percent >= self._last_log_percent + 10:
                        self.logger.info(f"{self.desc}: {percent}% ({self.n}/{self.total} {self.unit})")
                        self._last_log_percent = percent
                else:
                    # Log every 50 items if no total
                    if self.n % 50 == 0:
                        self.logger.info(f"{self.desc}: {self.n} {self.unit}")
                        
            def close(self):
                self.logger.info(f"Completed: {self.desc} ({self.n}/{self.total or self.n} {self.unit})")
                
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                self.close()
        
        return FileOnlyCounter(total, desc, unit, self.logger)
    
    def update_main_progress(self, n: int = 1):
        """Update the main progress bar"""
        if self._main_bar:
            self._main_bar.update(n)
    
    def log_info(self, message: str):
        """Log info message to file only"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message to file only"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message to file only"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message to file only"""
        self.logger.debug(message)
    
    def print_status(self, message: str):
        """Print status message to terminal above progress bars (use sparingly)"""
        if self.process_name == "MainProcess":
            # Use tqdm.write only for important status messages
            tqdm.write(f"ℹ️  {message}")
    
    def get_log_file_path(self) -> Path:
        """Get the current log file path"""
        return ProgressManager._log_file
    
    def cleanup(self):
        """Clean up all progress bars"""
        for bar in self._progress_bars.values():
            bar.close()
        self._progress_bars.clear()
        
        if self._main_bar:
            self._main_bar.close()
            self._main_bar = None


# Global singleton instance
def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance"""
    return ProgressManager()