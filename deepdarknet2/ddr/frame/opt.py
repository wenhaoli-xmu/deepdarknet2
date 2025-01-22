from .context import lib
from .errors import StopTraining, HessianNegtiveDefinite
import os
from scipy.io import savemat, loadmat
from threading import Lock, Thread, Semaphore


def _get_attr(obj, attr):
    attr = attr.split('.')
    for s in attr:
        obj = getattr(obj, s)
    return obj


def optimizer(fcn, **kwargs):
    method = kwargs.get('method', 'gd')

    if method == 'gd':
        return GradientDescent(fcn, **kwargs)
    elif method == 'newton':
        return NewtonsMethod(fcn, **kwargs)
    elif method == 'momentum':
        return Momentum(fcn, **kwargs)
    elif method == 'rmsprop':
        return RMSProp(fcn, **kwargs)
    elif method == 'adam':
        return Adam(fcn, **kwargs)
    elif method == 'adamw':
        return AdamW(fcn, **kwargs)
    else:
        raise ValueError(f'Optimize method {method} is unsupported.')


class CostFunction():
    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, 'value'):
            # 检查子类有没有value方法
            raise NotImplementedError
        if not hasattr(cls, 'gradient'):
            # 检查子类有没有gradient方法
            raise NotImplementedError
        if not hasattr(cls, 'hessian'):
            # 检查子类有没有hessian方法
            raise NotImplementedError
    
    def cache_value(self, value):
        self._value_cache = value
    
    def cache_gradient(self, grad):
        self._gradient_cache = grad

    def cache_hessian(self, hess):
        self._hessian_cache = hess


class Optimizer():
    def __init__(self, fcn, **kwargs):
        self._fcn = None
        self._x  = None

        self.fcn = fcn
        self.iter = 0
        self.max_iter = kwargs.get('max_iter', 1000)
        self.learning_rate = kwargs.get('learning_rate', 0.001)

        # 支持暂停以调整参数
        self.pause_key = kwargs.get('pause_key', 'p')
        self.key_thread = Thread(target=self.key_detect, daemon=True)
        self.mutex = Lock()
        self.sem1 = Semaphore(0)
        self.sem2 = Semaphore(0)
        self.pause = False
        
        # 与梯度检测有关的输入变量
        self.if_check_grads = kwargs.get('check_grads', False)
        self.grad_check_seq = kwargs.get('grad_check_seq', None)
        self.grad_check_disturb =kwargs.get('grad_check_disturb', 1e-4)
        self.if_check_hess = kwargs.get('check_hess', False)

        if self.grad_check_seq is not None:
            self.grad_check_seq = [int(elem) for elem in self.grad_check_seq]


    def key_detect(self):
        while True:
            inpu = input()
            if inpu == self.pause_key:
                # 检测到按下暂停键
                self.mutex.acquire()
                self.pause = True
                self.mutex.release()
                self.sem1.acquire()
                while True:
                    inpu = input(">>> ")
                    if inpu != 'p':
                        continue
                    assert isinstance(inpu, str)
                    splits = inpu.split(' ')
                    if splits[0] == 'change':
                        if len(splits) != 3:
                            print(f">>> 指令格式错误")
                            continue
                        # 修改某些属性
                        attr = splits[1]
                        value = splits[2]
                        if attr == 'max_iter':
                            value = int(value)
                        elif attr == 'learning_rate' or attr == 'lr':
                            value = float(value)
                        else:
                            print(f">>> 请输入合法的attr值")
                            continue
                        print(f">>> 已将{attr}修改为{value}")
                        setattr(self, attr, value)
                    elif splits[0] == 'save':
                        path = os.path.join(os.path.dirname(__file__), '..', '..',
                                            f'{self.iter}.mat')
                        path = os.path.abspath(path)
                        savemat(path, {'x': lib.get_cpu_array(self.x)})
                        print(f">>> 已保存在{path}中")
                    elif splits[0] == 'load':
                        iter = ''.join([splits[1].rstrip('.mat'), '.mat'])
                        path = os.path.join(os.path.dirname(__file__), '..', '..', iter)
                        param = loadmat(path)['x']
                        self.x = lib.array(param)
                        self._reset()
                        print(f">>> 加载完毕")
                    elif splits[0] == 'show':
                        if len(splits) == 1:
                            print(f">>> 请输入需要显示得attr")
                            continue
                        attr = splits[1]
                        if not _get_attr(self, attr):
                            print(f">>> 请输入合法的attr")
                            continue
                        print(f">>> {_get_attr(self, attr)}")
                    elif splits[0] == 'call':
                        if len(splits) == 2:
                            try: fcn = _get_attr(self, splits[1]); fcn()
                            except Exception as e: print(e)
                        else:
                            print(f">>> 参数数目不对")
                    else:
                        print(f">>> 不持支的指令{splits[0]}")
                self.mutex.acquire()
                self.pause = False
                self.mutex.release()
                self.sem2.release()


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if not isinstance(value, lib.ndarray):
            raise TypeError(f'Expect a lib.ndarray; got {type(value)}.')
        if value.shape[1] != 1:
            raise ValueError(f'Expect a column vector; got shape {type(value)}.')
        self._x = value

    @property
    def fcn(self):
        return self._fcn

    @fcn.setter
    def fcn(self, fcn):
        if not isinstance(fcn, CostFunction):
            raise TypeError(f'Expect a CostFunction; got {type(fcn)}.')
        self._fcn = fcn

    def check_gradient(self):
        numerical_grads = []

        if self.grad_check_seq is not None:
            seq = self.grad_check_seq
        else:
            seq = range(self._x.shape[0])
        for i in seq:
            tmpx_plus = self._x.copy()
            tmpx_minus = self._x.copy()
            tmpx_plus[i,:] += self.grad_check_disturb
            tmpx_minus[i,:] -= self.grad_check_disturb
            grad = self._fcn.value(tmpx_plus) - self._fcn.value(tmpx_minus)
            grad /= 2 * self.grad_check_disturb
            numerical_grads.append(grad)
        numerical_grads = lib.array(numerical_grads).reshape(-1, 1)

        return lib.hstack([self._fcn.gradient(self._x)[seq], numerical_grads])

    def check_hessian(self):
        numerical_hess = []
        size = self._x.shape[0]
        for i in range(size):
            for j in range(size):
                tmpx_plus = self._x.copy()
                tmpx_minus = self._x.copy()
                tmpx_plus[j,:] += 1e-8
                tmpx_minus[j,:] -= 1e-8
                gradient_fcn = lambda x: self._fcn.gradient(x)[i,:]
                hess = gradient_fcn(tmpx_plus) - gradient_fcn(tmpx_minus)
                hess /= 2e-8
                numerical_hess.append(hess)
        numerical_hess = lib.array(numerical_hess).reshape(size, size)
        squared_error = (numerical_hess - self._fcn.hessian(self._x))**2
        squared_error = float(lib.mean(squared_error))
        return squared_error

    def optimize(self, x0):
        self.x = x0
        self.key_thread.start()

        try:
            while self.iter <= self.max_iter:
                # 进行线程同步
                self.mutex.acquire()
                if self.pause:
                    self.mutex.release()
                    self.sem1.release()
                    self.sem2.acquire()
                else:
                    self.mutex.release()
                if hasattr(self._fcn, 'before_callback'):
                    # 查看是否有定义回调函数
                    self._fcn.before_callback(self._x)

                if self.if_check_grads == True:
                    # 进行梯度检查
                    compare = self.check_gradient()
                    if hasattr(self._fcn, 'after_gradient_check_callback'):
                        self._fcn.after_gradient_check_callback(self._x, compare)
                    
                if self.if_check_hess == True:
                    # 进行海塞矩阵检查
                    mse = self.check_gradient()
                    if hasattr(self._fcn, 'after_hessian_check_callbcak'):
                        self._fcn.after_hessian_check_callbcak(self._x, mse)

                # 更新参数
                self._update()
                self.iter += 1

                if hasattr(self._fcn, 'after_callback'):
                    # 查看是否有定义回调函数
                    self._fcn.after_callback(self._x)

        except StopTraining:
            pass
        except HessianNegtiveDefinite:
            pass
        except Exception as e:
            raise e

        return self._x

    def _update(self):
        raise NotImplementedError
    
    def _reset(self):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, fcn, **kwargs):
        super().__init__(fcn, **kwargs)

    def _reset(self):
        ...

    def _update(self):
        grads = self._fcn.gradient(self._x)
        self._x -= self.learning_rate * grads


class NewtonsMethod(Optimizer):
    def __init__(self, fcn, **kwargs):
        super().__init__(fcn, **kwargs)

    def _reset(self):
        ...

    def _update(self):
        grads = self.fcn.gradient(self._x)
        hess = self.fcn.hessian(self._x)
        den = grads.T.dot(hess).dot(grads)
        if den < 0:
            # 海塞矩阵不正定，算法不收敛
            raise HessianNegtiveDefinite
        elif den == 0:
            # 算法已经收敛
            raise StopTraining
        self._x -= grads.T.dot(grads) / den * grads


class Momentum(Optimizer):
    def __init__(self, fcn, **kwargs):
        super().__init__(fcn, **kwargs)
        self.velocity_decay = kwargs.get('velocity_decay', 0.9)
        self._reset()

    def _reset(self):
        self._velocity = None

    def _update(self):
        grads = self.fcn.gradient(self._x)
        if self._velocity is None:
            self._velocity = lib.zeros_like(self._x)
        self._velocity = (self.velocity_decay * self._velocity
                        - self.learning_rate * grads)
        self._x += self._velocity


class RMSProp(Optimizer):
    def __init__(self, fcn, **kwargs):
        super().__init__(fcn, **kwargs)
        self.rho = kwargs.get('rho', 0.99)
        self._reset()

    def _reset(self):
        self._scale = None  # 学习率放缩
    
    def _update(self):
        grads = self.fcn.gradient(self._x)
        if self._scale is None:
            self._scale = lib.zeros_like(grads) 
        self._scale = (self.rho * self._scale
                    + (1 - self.rho) * grads ** 2)
        grads /= lib.sqrt((1e-8 + self._scale))
        self._x -= self.learning_rate * grads


class Adam(Optimizer):
    def __init__(self, fcn, **kwargs):
        super().__init__(fcn, **kwargs)
        self.rho1 = kwargs.get('rho1', 0.9)    # 一阶矩衰减
        self.rho2 = kwargs.get('rho2', 0.999)  # 二阶矩衰减
        self._reset()

    def _reset(self):
        self._s = None  # 一阶矩变量
        self._r = None  # 二阶矩变量
        self._t = int(0)  # 步数

    def _update(self):
        grads = self.fcn.gradient(self._x)
        if self._s is None:
            self._s = lib.zeros_like(grads)
            self._r = lib.zeros_like(grads)
        self._t += 1
        self._s *= self.rho1 
        self._s += (1 - self.rho1) * grads
        self._r *= self.rho2  
        self._r += (1 - self.rho2) * grads ** 2
        s_ = self._s / (1 - self.rho1 ** self._t)
        r_ = self._r / (1 - self.rho2 ** self._t)
        self._x -= self.learning_rate * (s_ / lib.sqrt(r_ + 1e-8))


class AdamW(Optimizer):
    def __init__(self, fcn, **kwargs):
        super().__init__(fcn, **kwargs)
        self.rho1 = kwargs.get('rho1', 0.9)
        self.rho2 = kwargs.get('rho2', 0.999)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self._reset()

    def _reset(self):
        self._s = None
        self._r = None
        self._t = int(0)

    def _update(self):
        grads = self.fcn.gradient(self._x)
        if self._s is None:
            self._s = lib.zeros_like(grads)
            self._r = lib.zeros_like(grads)
        self._x *= (1 - self.weight_decay * self.learning_rate)
        self._t += 1
        self._s *= self.rho1 
        self._s += (1 - self.rho1) * grads
        self._r *= self.rho2  
        self._r += (1 - self.rho2) * grads ** 2
        s_ = self._s / (1 - self.rho1 ** self._t)
        r_ = self._r / (1 - self.rho2 ** self._t)
        self._x -= self.learning_rate * (s_ / lib.sqrt(r_ + 1e-8))
