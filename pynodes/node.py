import queue
from multiprocessing.synchronize import Event as EventClass
from collections import defaultdict
from typing import Callable, Any
import logging
import time

from .message import Message


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[1;31m" # Bold red
    }

    RESET = "\033[0m"

    def formatTime(self, record, datefmt=None):
        timestamp = time.strftime("%H:%M:%S", self.converter(record.created))
        milliseconds = int(record.msecs)
        return f"{timestamp}.{milliseconds:03d}"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")

        # Copy to avoid modifying the original shared LogRecord
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.node_name = f"\033[1m{record.node_name}{self.RESET}"

        return super().format(record)


class Node:
    name = "unnamed"

    def __init__(self, node_id: int, inbox: queue.Queue, outbound: queue.Queue, stop_event: EventClass):
        self.node_id: int = node_id
        self.inbox: queue.Queue = inbox
        self.outbound: queue.Queue = outbound
        self.stop_event: EventClass = stop_event

        self._callbacks = defaultdict(list)
        self._sequences = defaultdict(int)
        self.done: bool = False
        self.logger = self._create_logger()

    def _create_logger(self):
        logger = logging.getLogger(f"Node.{self.name}.{self.node_id}")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.StreamHandler()

            formatter = ColoredFormatter(
                "%(asctime)s [%(node_name)s][%(levelname)s] %(message)s"
            )

            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.propagate = False

        return logging.LoggerAdapter(logger, {"node_name": self.name})

    def setup(self):
        pass

    def advertise(self, topic: str) -> None:
        self.outbound.put(("advertise", self.node_id, topic))

    def subscribe(self, topic: str, callback: Callable):
        self._callbacks[topic].append(callback)
        self.outbound.put(("subscribe", self.node_id, topic))

        while not self.stop_event.is_set():
            try:
                event = self.inbox.get(timeout=0.5)
                if event[0] == "subscribed" and event[1] == topic:
                    return
                self._dispatch(event)
            except queue.Empty:
                continue

    def publish(self, topic: str, payload: Any) -> None:
        self._sequences[topic] += 1

        msg = Message.create(
            topic,
            payload,
            sequence=self._sequences[topic]
        )

        self.outbound.put(("publish", self.node_id, msg))

    def _dispatch(self, event):
        kind = event[0]

        if kind == "message":
            msg = event[1]

            for callback in self._callbacks[msg.topic]:
                try:
                    callback(msg)
                except Exception as e:
                    self.logger.exception(f"Callback error: {e}")

    def spin(self):
        self.setup()
        self.logger.info("Started")

        while not self.stop_event.is_set() and not self.done:
            try:
                event = self.inbox.get(timeout=0.2)
                self._dispatch(event)
            except queue.Empty:
                continue

        self.logger.info("Stopped")
