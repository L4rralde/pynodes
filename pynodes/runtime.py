import multiprocessing as mp
import threading
import time

from .broker import Broker


def _node_entry(
    node_cls,
    node_id,
    inbox,
    outbound,
    stop_event
) -> None:
    node = node_cls(
        node_id=node_id,
        inbox=inbox,
        outbound=outbound,
        stop_event=stop_event
    )

    node.spin()


class Runtime:
    def __init__(self, start_method: str="spawn"):
        self.ctx = mp.get_context(start_method)
        self.outbound = self.ctx.Queue()
        self.stop_event = self.ctx.Event()

        self.nodes = {}
        self.inboxes = {}
        self.processes = {}

        self.broker = Broker(
            self.outbound,
            self.inboxes
        )

        self.broker_thread = threading.Thread(
            target=self.broker.spin,
            args=(self.stop_event,),
            daemon=True
        )

    def add(self, node_cls) -> None:
        node_id = node_cls.name

        if node_id in self.nodes:
            raise ValueError(f"Duplicate node: {node_id}")

        self.nodes[node_id] = node_cls

    def start(self) -> None:
        self.broker_thread.start()

        for node_id, node_cls in self.nodes.items():
            inbox = self.ctx.Queue()

            self.inboxes[node_id] = inbox

            process = self.ctx.Process(
                target=_node_entry,
                args=(
                    node_cls,
                    node_id,
                    inbox,
                    self.outbound,
                    self.stop_event,
                ),
                name=node_id
            )

            self.processes[node_id] = process

            process.start()
        
        print("[Runtime] All nodes launched", flush=True)

    def spin(self, check_interval=0.5) -> None:
        try:
            while not self.stop_event.is_set():
                for node_id, process in self.processes.items():
                    if not process.is_alive():
                        print(
                            f"[Runtime] Node exited: {node_id} "
                            f"(exit code: {process.exitcode})",
                            flush=True,
                        )

                        self.shutdown()
                        return

                    time.sleep(check_interval)
        except KeyboardInterrupt:
            self.shutdown()

    def run(self) -> None:
        self.start()
        self.spin()

    def shutdown(self):
        if self.stop_event.is_set():
            return

        print("[Runtime] Shutting down", flush=True)

        self.stop_event.set()

        for process in self.processes.values():
            process.join(timeout=3)

        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

        self.broker_thread.join(timeout=2)

        for inbox in self.inboxes.values():
            inbox.close()

        self.outbound.close()

        print("[Runtime] Shutdown complete", flush=True)
