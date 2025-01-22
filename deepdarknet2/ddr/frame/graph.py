from threading import Semaphore, Thread, current_thread
from functools import lru_cache, wraps

from .context import lib
from .errors import NoCalcPath


def thread(function):

    @wraps(function)
    def wrapper(*args):
        th = Thread(target=function, args=args)
        return th
    return wrapper


@lru_cache(maxsize=16384)
def get_descendant(node):
    result = set()

    def _traverse(node):
        # 递归算法
        nonlocal result
        result.add(node)
        for son in node.sons:
            _traverse(son)

    _traverse(node)
    return result


@lru_cache(maxsize=16384)
def get_ancestor(node):
    result = set()

    def _traverse(node):
        # 递归算法
        nonlocal result
        result.add(node)
        for pa in node.pas:
            _traverse(pa)

    _traverse(node)
    return result


def flow_backward(subgraph: set, xs: set, ys: set, feed_dict: dict):
    pool = []
    # 清空互斥信号量列表，设置初始的need值
    for node in subgraph:
        node.mutex_to_acquire = [None,] * len(node.sons)
        node.mutex_to_release = [None,] * len(node.pas)
        node.need_grad = 1 if node in xs else 0

    for node in subgraph:
        # 给结点的梯度设置值
        if isinstance(node, Variable):
            node.release_grad()
            if node in ys:
                cond = node in feed_dict and feed_dict[node] is not None
                node.feed_grad(feed_dict[node]) if cond else node.feed_ones_grad_auto()
        # 为该结点及其父亲添加相应信号量
        for i, son in enumerate(node.sons):
            if son in subgraph:
                sem = Semaphore(0)
                node.mutex_to_acquire[i] = sem
                son.mutex_to_release[son.pas.index(node)] = sem
                son.need_grad += 1  # 每个双亲需要其值一次
        # 开辟一个线程进行计算
        pool.append(node.backward_computing_thread(subgraph))
    # 开启全部线程并等待运行结束
    for th in pool:
        th.start()
    for th in pool:
        th.join()


def flow_forward(subgraph: set, ys: set, xs: set, feed_dict: dict):
    pool = []
    # 清空互斥信号量列表，设置初始的need值
    for node in subgraph:
        node.mutex_to_acquire = [None,] * len(node.pas)
        node.mutex_to_release = [None,] * len(node.sons)
        node.need_value = 1 if node in ys else 0

    for node in subgraph:
        # 给变量做初始化工作
        node.release_all()
        if node in xs: node.feed(feed_dict[node])
        # 进行信号量的设置
        for i, pa in enumerate(node.pas):
            if pa in subgraph:
                sem = Semaphore(0)
                node.mutex_to_acquire[i] = sem
                pa.mutex_to_release[pa.sons.index(node)] = sem
                pa.need_value += 1
        # 开辟一个线程
        pool.append(node.forward_computing_thread(subgraph))
    # 开启全部线程并等待计算结束
    for th in pool:
        th.start()
    for th in pool:
        th.join()


class Node:
    def __init__(self, name):
        self.sons = list()
        self.pas = list()
        self.name = name

        # 用于线程同步
        self.mutex_to_release = []
        self.mutex_to_acquire = []

    def addpa(self, pa):
        self.pas += (pa, )

    def addson(self, son):
        self.sons += (son, )

    def forward_computing_thread(self, subgraph):
        raise NotImplementedError(f"抽象方法不可调用")

    def backward_computing_thread(self, subgraph):
        raise NotImplementedError(f"抽象方法不可调用")

    def __repr__(self):
        return f'{self.name}'


class Variable(Node):
    def __init__(self, shape=None, initializer=None, name='var',
                 retain_value=False):
        """
        Parameters
        ----------
        shape : 通常只有参数变量节点才需要指定shape
        
        initializer : 通常只有参数变量结点才需要制定initializer

        retain_value : 前向传播的时候不删除值, 用于设置常数
        """
        self.shape = shape
        self.init = initializer

        self.value = None
        self.grad = None

        # 标记位
        self.need_value = 0
        self.need_grad = 0
        self.retain_value = retain_value

        # 仅用于feed_ones_grad_auto
        self.forward_shape = None

        super().__init__(name)

    def sample(self):
        if self.init is None:
            raise ValueError(f'变量 `{self}` 未设置 `Initializer`.')
        cond = self.shape is not None and -1 not in self.shape
        if cond is False:
            raise ValueError(f"变量 `{self} 未设定 `shape` 属性`")
        return self.init.sample(self.shape)

    def feed(self, value):
        if self.shape is not None:
            # 有shape属性
            self.value = value.reshape(self.shape)
        else:
            # 没有shape属性
            self.value = value if self.value is None else self.value + value
        self.forward_shape = self.value.shape # 仅用于feed_ones_grad_auto

    def feed_ones_grad_auto(self):
        self.grad = lib.ones(self.forward_shape, dtype=lib.GLOBAL_DTYPE)

    def feed_grad(self, grad):
        self.grad = grad if self.grad is None else self.grad + grad

    def release_value(self):
        if not self.retain_value: self.value = None

    def release_grad(self):
        self.grad = None

    def release_all(self):
        if not self.retain_value: self.value = None
        self.grad = None

    @property
    def need_compute_value(self):
        return self.value is None

    @property
    def need_compute_grad(self):
        return self.grad is None

    @thread
    def forward_computing_thread(self, _):
        # 首先申请所有的mutex锁
        for mutex in self.mutex_to_acquire:
            if mutex: mutex.acquire()
        # 然后释放所有的mutex锁
        for mutex in self.mutex_to_release:
            if mutex: mutex.release()

    @thread
    def backward_computing_thread(self, _):
        # 首先申请所有的mutex锁
        for mutex in self.mutex_to_acquire:
            if mutex: mutex.acquire()
        # 然后释放所有的mutex锁
        for mutex in self.mutex_to_release:
            if mutex: mutex.release()


class Operation(Node):
    def __init__(self, op, name='op'):
        super().__init__(name)
        self.op = op
        # 标记位

    def release_cache(self):
        """释放op的cache"""
        self.op.cache = None

    def release_all(self):
        self.op.cache = None

    @thread
    def forward_computing_thread(self, subgraph):
        """
        前向传播需要operator结点的所有父亲值
        默认其每个subgraph中的operator的父亲都有值
        但每个operator的son不一定都需要计算, 只需要计算在
        subgraph中存在的那些sons
        """
        # 获取线程同步锁
        for mutex in self.mutex_to_acquire:
            if mutex: mutex.acquire()
        # 获取父亲们的参数
        args = [pa.value for pa in self.pas]

        # 测试代码
        # print(f"{current_thread().name.lstrip('Thread-')}", end='-')
        # print(f"{self.name:20s}\t:", end='')
        # for pa in self.pas:
        #     print(f"{pa}", end='')
        #     if pa.value is None:
        #         raise RuntimeError(f"进行`{self}`时缺少`{pa}`的值")
        #     print(f"{pa.value.shape}", end='\t')
        # print()

        for i, (son, mutex) in enumerate(zip(self.sons, self.mutex_to_release)):
            if son in subgraph:
                # 只计算所有位于subgraph中的孩子的值
                son.feed(self.op.forward(*args, index=i))
                # 释放线程同步锁
                mutex.release()
        # 计算完释放父节点的内存
        for pa in self.pas:
            # 默认所有的父亲的值都存在
            if pa.need_value > 0:
                pa.need_value -= 1
            if pa.need_value == 0:
                pa.release_value()


    @thread
    def backward_computing_thread(self, subgraph):
        """
        反向传播需要operator结点所有的孩子值
        默认其每个subgraph中的operator的孩子都有值
        但每个operator的pa不一定都需要计算, 只需要计算在
        subgraph中存在的那些pa
        """
        # 获取线程同步锁
        for mutex in self.mutex_to_acquire:
            if mutex: mutex.acquire()
        # 获取儿子们的grads
        grads = [son.grad for son in self.sons]

        # 测试代码
        # print(f"{current_thread().name.lstrip('Thread-')}", end='-')
        # print(f"{self.name:20s}\t:", end='')
        # for son in self.sons:
        #     print(f"{son}", end='')
        #     if son.grad is None:
        #         raise RuntimeError(f"进行`{self}`时缺少`{son}`的梯度")
        #     print(f"{son.grad.shape}", end='\t')
        # print()

        for i, (pa, mutex) in enumerate(zip(self.pas, self.mutex_to_release)):
            if pa in subgraph:
                # 计算所有父亲的grad
                pa.feed_grad(self.op.backward(*grads, index=i))
                # 释放线程同步锁
                mutex.release()
        # 计算完释放儿子结点内存
        for son in self.sons:
            if son.need_grad > 0:
                son.need_grad -= 1
            if son.need_grad == 0:
                son.release_grad()
        # 都计算完成后，释放前向传播的cache
        self.release_cache()


@lru_cache(maxsize=1024)
def get_subgraph(*nodes, split):
    subgraph_xs = set()
    subgraph_ys = set()
    for xi in nodes[:split]:
        subgraph_xs = subgraph_xs.union(get_descendant(xi))
    for yi in nodes[split:]:
        subgraph_ys = subgraph_ys.union(get_ancestor(yi))
    subgraph = subgraph_xs.intersection(subgraph_ys)

    if len(subgraph) == 0:
        raise NoCalcPath
    return subgraph


def eval(*y, feed_dict: dict):
    assert isinstance(feed_dict, dict)
    nodes = tuple(feed_dict) + y; split = len(feed_dict)
    subgraph = get_subgraph(*nodes, split=split)
    flow_forward(subgraph, set(y), set(feed_dict), feed_dict)


def diff(y: tuple, x: tuple, feed_dict: dict={}):
    assert isinstance(feed_dict, dict)
    nodes = x + y; split = len(x)
    subgraph = get_subgraph(*nodes, split=split)
    flow_backward(subgraph, set(x), set(y), feed_dict)
