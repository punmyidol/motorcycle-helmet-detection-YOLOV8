import zmq
import logging

logger = logging.getLogger(__name__)


class FrameSender:
    """
    ZeroMQ REQ client that sends JPEG frames to the cloud inference server
    and waits for an acknowledgement before proceeding.

    REQ/REP pattern guarantees ordering and provides natural back-pressure:
    the Jetson cannot flood the cloud faster than the cloud can process.
    """

    def __init__(
        self,
        cloud_ip: str,
        inference_port: int,
        send_timeout_ms: int = 2_000,
        recv_timeout_ms: int = 5_000,
    ) -> None:
        self._endpoint     = f"tcp://{cloud_ip}:{inference_port}"
        self._send_timeout = send_timeout_ms
        self._recv_timeout = recv_timeout_ms

        self._context = zmq.Context()
        self._socket  = self._make_socket(send_timeout_ms, recv_timeout_ms)

        logger.info("FrameSender connected to %s", self._endpoint)

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def send(self, jpeg_bytes: bytes) -> bool:
        """
        Send *jpeg_bytes* to the cloud server.

        Returns ``True`` on success, ``False`` if the server did not
        acknowledge within the configured timeout (frame is dropped).
        """
        try:
            self._socket.send(jpeg_bytes)
            ack = self._socket.recv_string()
            logger.debug("ACK received: %s", ack)
            return True
        except zmq.Again:
            logger.warning("Timeout waiting for ACK — dropping frame and reconnecting")
            self._reconnect()
            return False
        except zmq.ZMQError as exc:
            logger.error("ZMQ error: %s — reconnecting", exc)
            self._reconnect()
            return False

    def close(self) -> None:
        """Release ZeroMQ resources cleanly."""
        self._socket.close()
        self._context.term()
        logger.info("FrameSender closed")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _make_socket(self, send_timeout_ms: int, recv_timeout_ms: int) -> zmq.Socket:
        sock = self._context.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, send_timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        # Don't linger on close — discard unsent messages immediately
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._endpoint)
        return sock

    def _reconnect(self) -> None:
        """Destroy and recreate the REQ socket to clear a stuck state."""
        try:
            self._socket.close()
        except zmq.ZMQError:
            pass

        self._socket = self._make_socket(self._send_timeout, self._recv_timeout)
        logger.info("Reconnected to %s", self._endpoint)