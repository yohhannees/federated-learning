import threading
import queue
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os

from .logging import setup_logging


class CommunicationProtocol:
    """Implements a custom communication protocol for federated learning."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        async_mode: bool = False,
    ):
        """
        Initialize the communication protocol.

        Args:
            timeout: Timeout for client responses in seconds
            max_retries: Maximum number of retries for failed communications
            retry_delay: Delay between retries in seconds
            async_mode: Whether to use asynchronous communication
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.async_mode = async_mode

        # Setup logging
        self.logger = setup_logging(
            "communication",
            os.path.join(
                "logs",
                f'communication_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            ),
        )

        # Communication queues
        self.client_queues: Dict[int, queue.Queue] = {}
        self.server_queue = queue.Queue()

        # Thread locks
        self.client_locks: Dict[int, threading.Lock] = {}
        self.server_lock = threading.Lock()

        # Client status tracking
        self.client_status: Dict[int, Dict[str, Any]] = {}

        self.logger.info(f"Initialized communication protocol (async={async_mode})")

    def register_client(self, client_id: int):
        """
        Register a new client.

        Args:
            client_id: Unique identifier for the client
        """
        with self.server_lock:
            if client_id not in self.client_queues:
                self.client_queues[client_id] = queue.Queue()
                self.client_locks[client_id] = threading.Lock()
                self.client_status[client_id] = {
                    "last_seen": time.time(),
                    "active": True,
                    "retries": 0,
                }
                self.logger.info(f"Registered client {client_id}")

    def unregister_client(self, client_id: int):
        """
        Unregister a client.

        Args:
            client_id: ID of the client to unregister
        """
        with self.server_lock:
            if client_id in self.client_queues:
                del self.client_queues[client_id]
                del self.client_locks[client_id]
                del self.client_status[client_id]
                self.logger.info(f"Unregistered client {client_id}")

    def send_to_client(self, client_id: int, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific client.

        Args:
            client_id: ID of the target client
            message: Message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            with self.client_locks.get(client_id, threading.Lock()):
                if client_id in self.client_queues:
                    self.client_queues[client_id].put(message)
                    self.client_status[client_id]["last_seen"] = time.time()
                    self.logger.info(f"Sent message to client {client_id}")
                    return True
                else:
                    self.logger.warning(f"Client {client_id} not found")
                    return False
        except Exception as e:
            self.logger.error(f"Error sending message to client {client_id}: {str(e)}")
            return False

    def receive_from_client(
        self, client_id: int, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Receive a message from a specific client.

        Args:
            client_id: ID of the source client
            timeout: Optional timeout in seconds

        Returns:
            Received message or None if timeout/error
        """
        try:
            timeout = timeout or self.timeout
            with self.client_locks.get(client_id, threading.Lock()):
                if client_id in self.client_queues:
                    message = self.client_queues[client_id].get(timeout=timeout)
                    self.client_status[client_id]["last_seen"] = time.time()
                    self.logger.info(f"Received message from client {client_id}")
                    return message
                else:
                    self.logger.warning(f"Client {client_id} not found")
                    return None
        except queue.Empty:
            self.logger.warning(f"Timeout waiting for message from client {client_id}")
            return None
        except Exception as e:
            self.logger.error(
                f"Error receiving message from client {client_id}: {str(e)}"
            )
            return None

    def broadcast_to_clients(
        self, message: Dict[str, Any], exclude: Optional[List[int]] = None
    ) -> Dict[int, bool]:
        """
        Broadcast a message to all registered clients.

        Args:
            message: Message to broadcast
            exclude: Optional list of client IDs to exclude

        Returns:
            Dictionary mapping client IDs to success status
        """
        exclude = exclude or []
        results = {}

        with self.server_lock:
            for client_id in self.client_queues:
                if client_id not in exclude:
                    results[client_id] = self.send_to_client(client_id, message)

        return results

    def check_client_status(self, client_id: int) -> bool:
        """
        Check if a client is active and responsive.

        Args:
            client_id: ID of the client to check

        Returns:
            True if client is active, False otherwise
        """
        with self.server_lock:
            if client_id not in self.client_status:
                return False

            status = self.client_status[client_id]
            current_time = time.time()

            # Check if client has timed out
            if current_time - status["last_seen"] > self.timeout:
                status["active"] = False
                self.logger.warning(f"Client {client_id} timed out")
                return False

            return status["active"]

    def handle_client_dropout(
        self, client_id: int, max_retries: Optional[int] = None
    ) -> bool:
        """
        Handle client dropout with retry mechanism.

        Args:
            client_id: ID of the dropped client
            max_retries: Optional maximum number of retries

        Returns:
            True if client recovered, False otherwise
        """
        max_retries = max_retries or self.max_retries

        with self.server_lock:
            if client_id not in self.client_status:
                return False

            status = self.client_status[client_id]
            status["retries"] += 1

            if status["retries"] > max_retries:
                self.logger.warning(
                    f"Client {client_id} failed after {max_retries} retries"
                )
                self.unregister_client(client_id)
                return False

            # Wait before retry
            time.sleep(self.retry_delay)

            # Check if client recovered
            if self.check_client_status(client_id):
                status["retries"] = 0
                self.logger.info(f"Client {client_id} recovered")
                return True

            return False

    def get_active_clients(self) -> List[int]:
        """
        Get list of currently active clients.

        Returns:
            List of active client IDs
        """
        with self.server_lock:
            return [
                client_id
                for client_id, status in self.client_status.items()
                if status["active"]
            ]

    def cleanup_inactive_clients(self, timeout: Optional[float] = None):
        """
        Clean up inactive clients.

        Args:
            timeout: Optional timeout threshold in seconds
        """
        timeout = timeout or self.timeout
        current_time = time.time()

        with self.server_lock:
            inactive_clients = [
                client_id
                for client_id, status in self.client_status.items()
                if current_time - status["last_seen"] > timeout
            ]

            for client_id in inactive_clients:
                self.logger.warning(f"Cleaning up inactive client {client_id}")
                self.unregister_client(client_id)
