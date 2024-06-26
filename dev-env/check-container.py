"""Simple script to start and stop the storage service."""

import logging
import subprocess
import time
import urllib.request
from pathlib import Path

# Set up logging
logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
    level=logging.INFO,
)
logger = logging.getLogger("container-check")


def check_container(container_name: str = "freva-rest") -> None:
    """Check if the contianer starts up."""
    try:
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "--net=host",
                "-e",
                "MONGO_USER=mongo",
                "-e",
                "MONGO_PASSWORD=secret",
                "-e",
                "MONGO_HOST=localhost:27017",
                "-e",
                "API_PORT=7777",
                "-e",
                "API_WORKER=8",
                "-e",
                "MONGO_DB=search_stats",
                container_name,
            ],
        )
        time.sleep(10)
        if process.poll() is not None:
            raise RuntimeError("Container died.")
    except Exception as error:
        logger.critical("Strting the container failed: %s", error)
        raise
    process.terminate()
    logger.info("Container seems to work!")


if __name__ == "__main__":
    check_container()
