from .context import Variable


class Layer():
    def __init__(self, input_set=tuple(), param_set=tuple(), 
            output_set=tuple(), extra_input_set=tuple(), 
            extra_output_set=tuple()):
        assert isinstance(input_set, tuple)
        assert isinstance(param_set, tuple)
        assert isinstance(output_set, tuple)
        assert isinstance(extra_input_set, tuple)
        assert isinstance(extra_output_set, tuple)

        # 记录网络的输入、参数、输出变量集合
        self.param_set = param_set
        self.input_set = input_set
        self.extra_input_set = extra_input_set
        self.extra_output_set = extra_output_set
        self.output_set = output_set
    

def from_siso(op, *args, **kwargs):
    """仅支持单输入单输出"""
    input_ = Variable()
    result = op(input_, *args, **kwargs)
    return Layer(input_set=(input_,), output_set=(result,))


def from_mimo(op, input_count: int, output_count: int, *args, **kwargs):
    """支持多输入多输出"""
    inputs = tuple((Variable() for _ in range(input_count)))
    if output_count == 1:
        result = (op(*inputs, *args, **kwargs),)
    else:
        result = op(*inputs, *args, **kwargs)
    return Layer(input_set=inputs, output_set=result)
    

def setup(layer: Layer):
    """将Layer对象进行Setup, 构成网络的输入、输出和参数"""
    input_set = layer.input_set + layer.extra_input_set
    param_set = layer.param_set
    output_set = layer.extra_output_set + layer.output_set

    return input_set, param_set, output_set


def link_pairs(output: Variable, input: Variable):
    """
    Function
    ---
    Link the OUTPUT node to the INPUT node

    Scenes
    ---
    Establish links between EXTRA_INPUT_SET and EXTRA_OUTPUT_SET
    """
    assert isinstance(output, Variable)
    assert isinstance(input, Variable)
    for rear in input.sons:
        loc = rear.pas.index(input)
        rear.pas[loc] = output
        output.addson(rear)
    del input


def link_layers(front: Layer, rear: Layer):
    """将Layer1的ouptut_set连接到Layer2的input_set"""
    for output, input_ in zip(front.output_set, rear.input_set):
        link_pairs(output, input_)


class Sequence(Layer):
    def __init__(self, layers: tuple):
        param_set = tuple()
        extra_input_set = tuple()
        extra_output_set = tuple()

        # 将第一层的input_set加入input_set
        input_set = layers[0].input_set
        output_set = layers[-1].output_set
        
        for i in range(len(layers) - 1):
            front = layers[i]; rear = layers[i+1]
            # 并入集合
            extra_input_set += front.extra_input_set
            extra_output_set += front.extra_output_set
            param_set += front.param_set
            # 将layer与layer之间进行连接
            link_layers(front, rear)

        # 将最后一层也并入集合
        layer = layers[-1]
        extra_input_set += layer.extra_input_set
        extra_output_set += layer.extra_output_set
        param_set += layer.param_set

        # 初始化父类
        super().__init__(input_set=input_set,
            param_set=param_set,
            output_set=output_set,
            extra_input_set=extra_input_set,
            extra_output_set=extra_output_set,)


class Parallel(Layer):
    def __init__(self, layers):
        input_set = tuple()
        output_set = tuple()
        param_set = tuple()
        extra_input_set = tuple()
        extra_output_set = tuple()

        for layer in layers:
            input_set += layer.input_set
            output_set += layer.output_set
            param_set += layer.param_set
            extra_input_set += layer.extra_input_set
            extra_output_set += layer.extra_output_set

        super().__init__(
            input_set=input_set, param_set=param_set,
            output_set=output_set, extra_input_set=extra_input_set,
            extra_output_set=extra_output_set)


class Branch(Layer):
    def __init__(self, layer, layers, method='in_branch'):
        """用于连接分支结构"""

        parallel_layer = Parallel(layers)
        
        if method == 'in_branch':
            seq = Sequence((parallel_layer, layer))
        elif method == 'out_branch':
            seq = Sequence((layer, parallel_layer))
        else:
            raise ValueError(f"不支持 `{method}`.")

        super().__init__(input_set=seq.input_set,
            param_set=seq.param_set,
            output_set=seq.output_set,
            extra_input_set=seq.extra_input_set,
            extra_output_set=seq.extra_output_set)
