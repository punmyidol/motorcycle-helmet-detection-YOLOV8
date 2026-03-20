# shared/message_schema.py
#
# Single source of truth for every message exchanged between nodes.
#
# Node topology:
#
#   [Jetson]  ---(JPEG bytes)--->  [Cloud]  ---(label string)--->  [Alert Device]
#              <---( ACK string)---
#
# Import this module on every node instead of hardcoding strings/values.
# Python 2.7 compatible -- no enums, no dataclasses, no f-strings.

# =============================================================================
# CHANNEL 1: Jetson <-> Cloud  (zmq.REQ / zmq.REP)
# =============================================================================
#
# REQUEST  (Jetson -> Cloud)
#   Raw JPEG bytes.  No envelope, no header -- just the compressed frame.
#   Send with:  socket.send(jpeg_bytes)
#   Recv with:  jpeg_bytes = socket.recv()
#
# RESPONSE (Cloud -> Jetson)
#   A single ASCII string defined below.

# Cloud sends this string back to the Jetson after every frame it processes.
ACK = "ok"


# =============================================================================
# CHANNEL 2: Cloud -> Alert Device  (zmq.PUSH / zmq.PULL)
# =============================================================================
#
# The cloud sends one of three label strings after every inference pass.
# The alert device acts only on LABEL_NO_HELMET; the others are informational.

LABEL_HELMET    = "helmet"      # rider is wearing a helmet
LABEL_NO_HELMET = "no-helmet"   # rider is NOT wearing a helmet  -> trigger alert
LABEL_UNKNOWN   = "unknown"     # no motorcycle / helmet detected in frame

# Convenience set -- use for membership tests instead of repeating literals:
#   if label not in ALL_LABELS: ...
ALL_LABELS = (LABEL_HELMET, LABEL_NO_HELMET, LABEL_UNKNOWN)


# =============================================================================
# HELPERS
# =============================================================================

def is_valid_label(label):
    """Return True if label is one of the three recognised detection strings."""
    return label in ALL_LABELS


def should_alert(label):
    """Return True if the label should trigger the alert device."""
    return label == LABEL_NO_HELMET