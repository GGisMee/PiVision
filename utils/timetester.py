import time
import functools
import os
import sys

from rich.console import Console
from rich.table import Table

console = Console()
from typing import Union

class TimeTester:
    _instance = None  # Static variable to store the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TimeTester, cls).__new__(cls)
            cls._instance._init_internal(*args, **kwargs)
        return cls._instance

    def _init_internal(self, enabled=True, sample_each_and_every: float = 1, print_at_run:bool = False):
        '''A custom class to time functions across multiple modules.'''
        self.enabled = enabled
        self.data = {}  # Stores execution times
        self.just_mean = False  # Default behavior
        self.print_at_run: bool = print_at_run
        self.current_mean_values = {}
        self.console = Console()

        self.measure_between_vals = {}

    def testtime(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if self.print_at_run:
                print(f"[INFO] Timing for {func.__name__}: {elapsed_time:.6f} seconds")  # Debug print to check timing

            if self.just_mean:
                self._set_new_mean(func.__name__, elapsed_time)

            return result
        return wrapper

    def print_execution_times(self):
        print("Execution times of functions:")
        for func_name, data in self.current_mean_values.items():
            avg_time = data['t']
            print(f"Function {func_name} took {avg_time:.6f} seconds on average.")

    def set_just_mean(self, max_values: int = 20):
        '''Only stores the mean execution time for each function.'''
        self.just_mean = True
        self.max_values = max_values
        self.current_mean_values = {}

    def _set_new_mean(self, func_name: str, elapsed_time: float):
        if func_name not in self.current_mean_values:
            self.current_mean_values[func_name] = {'t': elapsed_time, 'n': 1}
        else:
            n = self.current_mean_values[func_name]['n']
            self.current_mean_values[func_name]['t'] = (n * self.current_mean_values[func_name]['t'] + elapsed_time) / (n + 1)
            
        if n+1 != self.max_values:
            self.current_mean_values[func_name]['n'] += 1

    def measure_between(self, identifier:Union[str, int]):

        # if no value yet, or old removed value, add new value
        if (identifier not in self.measure_between_vals.keys()) or (not self.measure_between_vals[identifier]):
            self.measure_between_vals[identifier] = time.perf_counter()
        else: # if the second value: print and remove old value
            time_elapsed = time.perf_counter()-self.measure_between_vals[identifier]
            print(f"[INFO] Timing for {identifier}: {time_elapsed:.6f} seconds") 
            self.measure_between_vals[identifier] = None


    def print_execution_times(self):
        """Displays execution times using a rich table."""
        if not self.current_mean_values:
            self.console.print("[bold red]No timing data recorded yet![/bold red]")
            return

        table = Table(title="Function Execution Times", show_lines=True)
        table.add_column("Function", style="cyan", justify="left")
        table.add_column("Avg Time (s)", style="green", justify="right")
        table.add_column("Calls", style="magenta", justify="right")

        for func_name, data in self.current_mean_values.items():
            table.add_row(func_name, f"{data['t']:.6f}", str(data["n"]))

        self.console.print(table)

# Create a shared instance
timetester = TimeTester(print_at_run= True)
