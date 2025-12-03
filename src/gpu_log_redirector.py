import sys
import os
import datetime

class DualLogger:
    """
    A class to redirect stdout (print statements) to both the console 
    and a specified log file.
    
    Usage:
    log_file_path = 'training_run.log'
    with DualLogger(log_file_path):
        print("This message goes to both console and file.")
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = None
        self.filepath = filepath
        self.is_open = False

    def __enter__(self):
        """Opens the log file and redirects sys.stdout."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            self.log_file = open(self.filepath, 'a', buffering=1, encoding='utf-8')
            sys.stdout = self
            self.is_open = True
            # Log the start time
            self.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING TRAINING RUN ---\n")
        except Exception as e:
            self.terminal.write(f"Error initializing DualLogger: {e}\n")
            sys.stdout = self.terminal # Restore stdout immediately on failure
            self.is_open = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restores sys.stdout and closes the log file."""
        if self.is_open:
            self.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- TRAINING RUN ENDED ---\n")
            sys.stdout = self.terminal # Restore terminal output
            self.log_file.close()

    def write(self, message):
        """Writes the message to both the terminal and the log file."""
        self.terminal.write(message)
        if self.is_open and self.log_file:
            self.log_file.write(message)

    def flush(self):
        """Ensures the stream is flushed (required for file streams)."""
        self.terminal.flush()
        if self.is_open and self.log_file:
            self.log_file.flush()