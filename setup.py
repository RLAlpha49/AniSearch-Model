from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        # Install PyTorch with CUDA support
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ]
        )


setup(
    name="AniSearchModel",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "transformers",
        "sentence-transformers",
        "tqdm",
        "datasets",
        "flask",
        "flask-limiter",
        "waitress",
        "gunicorn",
        "pytest",
        "pytest-order",
        "flask_limiter",
    ],
    cmdclass={
        "install": PostInstallCommand,
    },
    entry_points={
        "console_scripts": [
            "run-server=src.run_server:run_server",
        ],
    },
)
