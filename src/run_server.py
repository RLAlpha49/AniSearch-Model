"""
Starts the Flask application using an appropriate server based on the operating system.

The module accepts optional command line arguments:
    - First argument: Device type ('cuda' or 'cpu', defaults to 'cpu')
    - Second argument: Number of workers/threads (positive integer, defaults to 4)

Server selection:
    - Linux: Uses Gunicorn with specified number of worker processes
    - Windows: Uses Waitress with specified number of threads
    - Other OS: Uses Flask's built-in development server

The server runs on port 21493 and binds to all network interfaces (0.0.0.0).
"""

import platform
import subprocess
import sys
import os


def run_server() -> None:
    """
    Start the Flask application using an OS-appropriate server with configurable settings.

    Command line arguments:

        argv[1]: Device type ('cuda' or 'cpu', defaults to 'cpu')

        argv[2]: Number of workers/threads (positive integer, defaults to 4)

    Environment variables set:

        DEVICE: Set to the specified device type ('cuda' or 'cpu')

    Server configuration:

    Linux: Gunicorn
        - Workers: Specified by argv[2]
        - Logs: ./logs/gunicorn_access.log and gunicorn_error.log
        - Binds to: 0.0.0.0:21493

    Windows: Waitress
        - Threads: Specified by argv[2]
        - Port: 21493

        Other OS: Flask development server
    """
    os_type: str = platform.system()

    # Check for device argument
    if len(sys.argv) > 1:
        device = sys.argv[1].lower()
        if device not in ["cuda", "cpu"]:
            print("Invalid device argument. Use 'cuda' or 'cpu'.")
            sys.exit(1)
    else:
        device = "cpu"  # Default to CPU if no argument is provided

    # Check for threads argument
    if len(sys.argv) > 2:
        try:
            threads = int(sys.argv[2])
            if threads < 1:
                raise ValueError
        except ValueError:
            print("Invalid threads argument. Use a positive integer.")
            sys.exit(1)
    else:
        threads = 4  # Default to 4 threads if no argument is provided

    # Set the device as an environment variable
    os.environ["DEVICE"] = device

    if os_type == "Linux":
        # Use Gunicorn on Linux
        print(f"Running on Linux. Starting Gunicorn server with {threads} workers.")
        subprocess.run(
            [
                "gunicorn",
                "-w",
                str(threads),
                "-b",
                "0.0.0.0:21493",
                "--access-logfile",
                "./logs/gunicorn_access.log",
                "--error-logfile",
                "./logs/gunicorn_error.log",
                "src.api:app",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": "."},
        )
    elif os_type == "Windows":
        # Use Waitress on Windows
        print(f"Running on Windows. Starting Waitress server with {threads} threads.")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "waitress",
                "--port=21493",
                f"--threads={threads}",
                "src.api:app",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": "."},
        )
    else:
        print(f"Running on {os_type}. Using Flask's built-in server.")
        subprocess.run(["python", "src.api:app"], check=True)


if __name__ == "__main__":
    run_server()
