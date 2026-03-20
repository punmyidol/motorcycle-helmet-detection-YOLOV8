import zmq

from config import CLOUD_IP, INFERENCE_PORT, SEND_TIMEOUT_MS


class FrameSender(object):
    """
    ZeroMQ PUSH socket wrapper.

    Sends JPEG bytes to the cloud inference server and waits for an ACK.
    Uses REQ/REP pattern (not PUSH/PULL) so the Jetson knows the cloud
    received the frame before sending the next one -- natural back-pressure.

    Handles reconnection transparently: if the cloud goes away and comes
    back, ZMQ will re-establish the connection without any extra code here.
    """

    def __init__(self):
        self._context = zmq.Context()
        self._socket  = None
        self._connect()

    def send(self, jpeg_bytes):
        """
        Send one JPEG frame and block until ACK is received.

        Parameters
        ----------
        jpeg_bytes : bytes
            Encoded frame from Preprocessor.process().

        Returns
        -------
        bool
            True on success, False on timeout or error.
        """
        try:
            self._socket.send(jpeg_bytes)
            # Wait for ACK from cloud server ("ok")
            ack = self._socket.recv_string()
            return ack == "ok"
        except zmq.Again:
            print("[Sender] WARNING: send/recv timed out, dropping frame")
            # Reconnect to clear the stuck REQ state machine
            self._reconnect()
            return False
        except zmq.ZMQError as e:
            print("[Sender] ERROR: %s" % str(e))
            self._reconnect()
            return False

    def close(self):
        """Release ZMQ resources cleanly."""
        if self._socket:
            self._socket.close()
        self._context.term()

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _connect(self):
        self._socket = self._context.socket(zmq.REQ)
        # Timeouts prevent a dead cloud server from hanging the Jetson forever
        self._socket.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT_MS)
        self._socket.setsockopt(zmq.RCVTIMEO, SEND_TIMEOUT_MS)
        # Drop pending messages immediately on close (don't block shutdown)
        self._socket.setsockopt(zmq.LINGER, 0)
        endpoint = "tcp://%s:%d" % (CLOUD_IP, INFERENCE_PORT)
        self._socket.connect(endpoint)
        print("[Sender] Connected to %s" % endpoint)

    def _reconnect(self):
        print("[Sender] Reconnecting...")
        try:
            self._socket.close()
        except Exception:
            pass
        self._connect()