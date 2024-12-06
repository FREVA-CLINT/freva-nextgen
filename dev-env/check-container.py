import logging
import subprocess
import time

# Set up logging
logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
    level=logging.INFO,
)
logger = logging.getLogger("container-check")


def check_container(image_name: str = "freva-rest", container_name: str = "freva-rest") -> None:
    """Check if the contianer starts up."""
    try:
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "--name", container_name,
                "--net=host",
                "-e",
                "API_MONGO_USER=mongo",
                "-e",
                "API_MONGO_PASSWORD=secret",
                "-e",
                "API_MONGO_HOST=localhost:27017",
                "-e",
                "API_PORT=7777",
                "-e",
                "API_WORKER=8",
                "-e",
                "API_MONGO_DB=search_stats",
                "-e",
                "API_OIDC_DISCOVERY_URL=http://localhost:8080/realms/freva/.well-known/openid-configuration",
                "-e",
                "USE_MONGODB=1",
                "-e",
                "USE_SOLR=1",
                image_name,
            ],
        )
        time.sleep(20)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Container died. Exit code: {process.returncode}. "
                             f"Stdout: {stdout}, Stderr: {stderr}")
        
        logger.info("Container started successfully.")
    except Exception as error:
        logger.critical("Starting the container failed: %s", error)
        raise
    finally:
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        try:
            logger.info(f"Stopping container {container_name}")
            stop_process = subprocess.Popen(
                ["docker", "stop", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stop_process.wait(timeout=15)
            
            logger.info(f"Removing container {container_name}")
            rm_process = subprocess.Popen(
                ["docker", "rm", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rm_process.wait(timeout=10)
            
            logger.info("Container cleanup completed successfully")
        except Exception as cleanup_error:
            logger.error("Failed to clean up container: %s", cleanup_error)


if __name__ == "__main__":
    check_container()
