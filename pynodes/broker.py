import queue
from collections import defaultdict


class Broker:
    def __init__(self, outbound: queue.Queue, inboxes):
        self.outbound = outbound
        self.inboxes = inboxes

        self.subscribers = defaultdict(set)
        self.publishers = defaultdict(set)

        self.running = False

    def handle(self, event):
        kind = event[0]
        node_id = event[1]

        if kind == "subscribe":
            topic = event[2]
            self.subscribers[topic].add(node_id)

            self.inboxes[node_id].put(
                ("subscribed", topic)
            )

            print(
                f"[Broker] {node_id} subscribed to {topic}",
                flush=True
            )

        elif kind == "advertise":
            topic = event[2]
            self.publishers[topic].add(node_id)
            print(
                f"[Broker] {node_id} advertises {topic}",
                flush=True,
            )

        elif kind == "publish":
            msg = event[2]
            topic = msg.topic

            for subscriber in list(self.subscribers[topic]):
                inbox = self.inboxes.get(subscriber)

                if inbox is not None:
                    inbox.put(("message", msg))

        else:
            raise RuntimeError("[Broker] kind: {kind} not recognized")


    def spin(self, stop_event):
        self.running = True
        while not stop_event.is_set():
            try:
                event = self.outbound.get(timeout=0.2)
                self.handle(event)
            except queue.Empty:
                continue

        self.running = False
