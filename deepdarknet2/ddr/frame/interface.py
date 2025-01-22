from .graph import Variable, Operation
from .context import ops


def _link(pa, son):
    pa.addson(son)
    son.addpa(pa)


def unflatten(input: Variable, dim: int, split: tuple):
    op = Operation(op=ops.UnFlatten(dim, split),
                   name=f'unflatten({dim}){split}')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def flatten(input: Variable, start_dim: int, end_dim: int):
    op = Operation(op=ops.Flatten(start_dim=start_dim, end_dim=end_dim),
                   name=f'flatten({start_dim})')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def split(input: Variable, axis, start, stop, step=1):
    op = Operation(op=ops.Split(axis=axis, start=start, stop=stop, step=step),
                   name='split')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def tile(input: Variable, rep):
    op = Operation(op=ops.Tile(rep=rep), name=f"tile{rep}")
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def transpose(input: Variable, axes):
    op = Operation(op=ops.Transpose(*axes),name=f"transpose{axes}")
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def deconcatenate(input: Variable, axis, split):
    op = Operation(op=ops.Deconcatenate(axis=axis, plan=split),
                   name=f"deconcat({axis}){split}")
    output = []
    _link(input, op)
    for _ in range(len(split)):
        o = Variable()
        _link(op, o)
        output.append(o)
    return output


def concatenate(*input, axis):
    op = Operation(op=ops.Concatenate(axis=axis),
                   name=f'concat({axis})')
    output = Variable()
    for i in input:
        _link(i, op)
    _link(op, output)
    return output


def reshape(input, shape):
    op = Operation(op=ops.Reshape(*shape),
                   name=f"reshape{shape}")
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def matmul(input1, input2):
    op = Operation(op=ops.Matmul(), name='@')
    output = Variable()
    _link(input1, op); _link(input2, op)
    _link(op, output)
    return output


def bmm(input1, input2):
    op = Operation(op=ops.Bmm(), name='bmm')
    output = Variable()
    _link(input1, op); _link(input2, op)
    _link(op, output)
    return output


def dot(input1, input2):
    op = Operation(op=ops.Dot(), name='*')
    output = Variable()
    _link(input1, op); _link(input2, op)
    _link(op, output)
    return output


def plus(*inputs):
    op = Operation(op=ops.Plus(), name='+')
    output = Variable()
    for inpu in inputs:
        _link(inpu, op)
    _link(op, output)
    return output


def minus(input1, input2):
    op = Operation(op=ops.Minus(), name='-')
    output = Variable()
    _link(input1, op)
    _link(input2, op)
    _link(op, output)
    return output


def divide(input1, input2):
    op = Operation(op=ops.Divide(), name='/')
    output = Variable()
    _link(input1, op); _link(input2, op)
    _link(op, output)
    return output


def power(input, power):
    op = Operation(op=ops.Power(power=power), name='^')
    output = Variable()
    _link(input, op); _link(op, output)


def abs(input):
    op = Operation(op=ops.Abs(), name='abs')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def neg(input):
    op = Operation(op=ops.Neg(), name='neg')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def max(input, axis):
    """axis仅支持[0,1]"""
    assert axis in (0,1)
    op = Operation(op=ops.Max(axis=axis), name=f'max({axis})')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def min(input, axis):
    """axis仅支持[0,1]"""
    assert axis in (0,1)
    op = Operation(op=ops.Min(axis=axis), name=f'min{axis})')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def sum(input, axis):
    """目前仅支持2维的input, axis∈{0,1}"""
    op = Operation(op=ops.Sum(axis=axis), name=f'sum({axis})')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def exp(input):
    op = Operation(op=ops.Exp(), name='exp')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def log(input, fix: bool = True):
    op = Operation(op=ops.Log(fix=fix), name='log')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def batch_norm_1d(input, sample_axis):
    op = Operation(op=ops.BatchNorm1D(sample_axis=sample_axis),
                   name='bn')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def layer_norm(input, sample_axis=0):
    if sample_axis != 0:
        raise ValueError(f"layer_norm: `sample_axis` should be 0")
    op = Operation(op=ops.LayerNorm(), name='ln')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def l1_penalty(input, lamb):
    op = Operation(op=ops.L1Penalty(lamb=lamb),
                   name='l1')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def l2_penalty(input, lamb):
    op = Operation(op=ops.L2Penalty(lamb=lamb),
                   name='l2')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def dropout2d(input, feature_axis, keep_prob=0.5):
    op = Operation(op=ops.Dropout2D(feature_axis=feature_axis, keep_prob=keep_prob),
                   name=f'dropout2d({keep_prob})')
    output = Variable()
    _link(input, op); _link(op, output)
    return output

def dropout(input, keep_prob=.5):
    op = Operation(op=ops.DropoutElemWise(keep_prob=keep_prob),
                   name=f'dropout({keep_prob})')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def sae(input1, input2, sample_axis):
    op = Operation(op=ops.SumAbsoluteError(sample_axis),
                   name='sae')
    output = Variable()
    _link(input1, op); _link(input2, op); _link(op, output)
    return output


def mae(input1, input2, sample_axis):
    op = Operation(op=ops.MeanAbsoluteError(sample_axis=sample_axis),
                   name='mae')
    output = Variable()
    _link(input1, op); _link(input2, op); _link(op, output)
    return output


def mse(input1, input2, sample_axis):
    op = Operation(op=ops.MeanSquaredError(sample_axis=sample_axis),
                   name='mse')
    output = Variable()
    _link(input1, op); _link(input2, op); _link(op, output)
    return output


def cross_entropy(input1, input2, feature_axis, is_reduced=False):
    op = Operation(op=ops.CrossEntropy(feature_axis=feature_axis, 
        is_reduced=is_reduced), name='cross_entropy')
    output = Variable()
    _link(input1, op); _link(input2, op); _link(op, output)
    return output


def padding(input, height, width):
    op = Operation(op=ops.Padding(height=height, width=width),
                   name='padding')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def conv2d(image, kernel, strides=(1,1), padding=(0,0)):
    op = Operation(op=ops.Conv2D(strides=strides, padding=padding),
                   name=f'conv2d{strides}{padding}')
    output = Variable()
    _link(image, op); _link(kernel, op); _link(op, output)
    return output


def adaptive_pooling(image, obj_grid, method='max'):
    op = Operation(op=ops.AdaptivePooling(obj_grid, method),
                   name=f'adaptive pooling{obj_grid}')
    output = Variable()
    _link(image, op); _link(op, output)
    return output


def pooling(image, window_shape, strides, method='max', auto_padding=False):
    op = Operation(op=ops.Pooling(window_shape=window_shape, strides=strides,
                                  method=method, auto_padding=auto_padding,),
                   name=f'pooling{window_shape}{strides}')
    output = Variable()
    _link(image, op); _link(op, output)
    return output


def batch_norm_2d(input):
    op = Operation(op=ops.BatchNorm2D(), name='bn2d')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def relu(input):
    op = Operation(op=ops.Relu(), name='relu')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def gelu(input):
    op = Operation(op=ops.Gelu(), name='gelu')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def softplus(input):
    op = Operation(op=ops.Softplus(), name='softplus')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def sigmoid(input):
    op = Operation(op=ops.Sigmoid(), name='sigmoid')
    output = Variable()
    _link(input, op); _link(op, output)
    return output


def softmax(input, feature_axis, prevent_overflow=True):
    if prevent_overflow:
        input = minus(input, max(input, axis=feature_axis))
    op = Operation(op=ops.Softmax(feature_axis=feature_axis),
                   name='softmax')
    output = Variable()
    _link(input, op); _link(op, output)
    return output
