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


def run_server():
    """
    Determine the operating system and start the Flask application using the appropriate server.

    - On Linux, the application is run using Gunicorn with 4 worker processes.
    - On Windows, the application is run using Waitress.
    - On other operating systems, the application defaults to using Flask's built-in server.

    This function uses the `subprocess.run` method to execute the server command.
    """
    os_type = platform.system()

    if os_type == "Linux":
        # Use Gunicorn on Linux
        print("Running on Linux. Starting Gunicorn server.")
        subprocess.run(
            ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api:app"], check=True
        )
    elif os_type == "Windows":
        # Use Waitress on Windows
        print("Running on Windows. Starting Waitress server.")
        subprocess.run(["waitress-serve", "--port=5000", "api:app"], check=True)
    else:
        print(f"Running on {os_type}. Using Flask's built-in server.")
        subprocess.run(["python", "api.py"], check=True)


if __name__ == "__main__":
    run_server()
