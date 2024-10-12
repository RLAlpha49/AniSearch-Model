"""
This module determines the operating system and starts the Flask application
using the appropriate server.

- On Linux, it uses Gunicorn with 4 worker processes.
- On Windows, it uses Waitress.
- On other operating systems, it defaults to using Flask's built-in server.

The script uses the `subprocess.run` method to execute the server command.
"""

import platform
import subprocess
import sys
import os


def run_server() -> None:
    """
    Determine the operating system and start the Flask application using the appropriate server.

    - On Linux, the application is run using Gunicorn with 4 worker processes.
    - On Windows, the application is run using Waitress.
    - On other operating systems, the application defaults to using Flask's built-in server.

    This function uses the `subprocess.run` method to execute the server command.
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

    # Set the device as an environment variable
    os.environ["DEVICE"] = device

    if os_type == "Linux":
        # Use Gunicorn on Linux
        print("Running on Linux. Starting Gunicorn server.")
        subprocess.run(
            ["gunicorn", "-w", "4", "-b", "0.0.0.0:21493", "src.api:app"], check=True
        )
    elif os_type == "Windows":
        # Use Waitress on Windows
        print("Running on Windows. Starting Waitress server.")
        subprocess.run(["waitress-serve", "--port=21493", "src.api:app"], check=True)
    else:
        print(f"Running on {os_type}. Using Flask's built-in server.")
        subprocess.run(["python", "src.api:app"], check=True)


if __name__ == "__main__":
    run_server()
