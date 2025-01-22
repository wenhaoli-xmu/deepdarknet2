from threading import Thread, Semaphore
from queue import Queue


def worker(queue):
    while True:
        item = queue.get()
        fcn, subgraph = item
        fcn(subgraph)
        queue.task_done()


class ThreadPool():
    def __init__(self, max_workers=1024):
        self.queue = Queue(maxsize=max_workers)
        self.pool = [Thread(target=worker,
                            args=(self.queue,), daemon=True)
                     for _ in range(max_workers)]
        for th in self.pool: th.start()

    def submit(self, fcn, subgraph):
        item = (fcn, subgraph)
        self.queue.put(item)

    def wait(self):
        self.queue.join()
