"""Pipeline package: sliding window, smoothing, event bus, session logging,
recipe loading, and the engine that wires camera -> models -> state machine.

The UI consumes ONLY typed events (src/pipeline/events.py); no logic lives
in the front end.
"""
