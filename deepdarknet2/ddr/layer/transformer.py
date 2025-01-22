from .context import (
    Variable, initializer, lib,
    conv2d, reshape, transpose, matmul, bmm, softmax, divide, dropout,
    gelu, layer_norm, plus, dot, tile,
)

from .layer import Layer, Sequence, from_siso, setup, link_pairs
from .fcn import Linear


def _make_tuple(x):
    if not isinstance(x, (list, tuple)):
        return (x, x)
    return x


class PatchEmbedding(Layer):
    def __init__(self, img_size, patch_size, dim_token,
                 kernel_init=initializer.Gaussian(),
                 bias_init=initializer.Constant()):
        """
        参数
        ---
        img_size : int或tuple, 表示图片的尺寸
        patch_size : int或tuple, 表示patch的大小
        dim_token : int
        """

        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        num_token = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        
        image = Variable()
        kernel = Variable(shape=(3,patch_size[0],patch_size[1],dim_token),
                          initializer=kernel_init)
        bias = Variable(shape=(1,dim_token,1,1), initializer=bias_init)
        result = conv2d(image, kernel, 
                        strides=(patch_size[0],patch_size[1]),
                        padding=(0,0))
        result = plus(result, bias)
        result = reshape(result, (-1, dim_token, num_token))
        result = transpose(result, axes=(0,2,1))

        super().__init__(
            input_set=(image,),
            param_set=(kernel,bias),
            output_set=(result,))
        

class ConcatClassToken(Layer):
    def __init__(self, batch_size, dim_token):
        cls_token = Variable(name='cls_token', shape=(1,1,dim_token),
                             initializer=initializer.Constant(0))
        x = Variable()
        tmp = tile(cls_token, (batch_size,1,1))
        result = plus(tmp, x)

        super().__init__(
            input_set=(x,),
            param_set=(cls_token,),
            output_set=(result,))

        
class PosEmbedding(Layer):
    def __init__(self, num_token, dim_token, keep_prob):
        x = Variable()
        pos = Variable(name='pos_emb', shape=(1,num_token,dim_token),
                       initializer=initializer.Gaussian(0,1))
        res = dropout(plus(x, pos), keep_prob=keep_prob)

        super().__init__(
            input_set=(x,),
            param_set=(pos,),
            output_set=(res,))


class DotProdMultiHeadAttention(Layer):
    def __init__(self, num_heads, p_o, p_qk, p_v, n_q, n_kv, d_qk, d_v, 
                 keep_prob=1, kernel_init=initializer.Gaussian()):
        query = Variable() # shape of Q: (m, n_q, d_qk)
        key = Variable() # shape of K: (m, n_kv, d_qk)
        value = Variable() # shape of V: (m, n_kv, d_v)

        wq = Variable(name='attention.wq',shape=(d_qk, num_heads*p_qk), 
                      initializer=kernel_init)
        wk = Variable(name='attention.wk',shape=(d_qk, num_heads*p_qk), 
                      initializer=kernel_init)
        wv = Variable(name='attention.wv',shape=(d_v, num_heads*p_v), 
                      initializer=kernel_init)
        wo = Variable(name='attention.wo',shape=(num_heads*p_v, p_o), 
                      initializer=kernel_init)

        q = matmul(query, wq)
        k = matmul(key, wk)
        v = matmul(value, wv)

        q = reshape(q,(-1,n_q,num_heads,p_qk))
        k = reshape(k,(-1,n_kv,num_heads,p_qk))
        v = reshape(v,(-1,n_kv,num_heads,p_v))
        
        q = transpose(q, (0,2,1,3))
        k = transpose(k, (0,2,1,3))
        v = transpose(v, (0,2,1,3))

        q = reshape(q, (-1,n_q,p_qk))
        k = reshape(k, (-1,n_kv,p_qk))
        v = reshape(v, (-1,n_kv,p_v))

        const = Variable(shape=(1,1,1), retain_value=True)
        const.feed(lib.array((lib.sqrt(p_qk),),dtype=lib.GLOBAL_DTYPE))

        weights = bmm(q,transpose(k, (0,2,1)))
        weights = divide(weights,const)
        weights = softmax(weights,feature_axis=2)
        if keep_prob < 1:
            weights = dropout(weights,keep_prob=keep_prob)

        attention = bmm(weights,v)
        
        attention = reshape(attention, (-1,num_heads,n_q,p_v))
        attention = transpose(attention, (0,2,1,3))
        attention = reshape(attention, (-1,n_q,num_heads*p_v))
        result = matmul(attention,wo)

        super().__init__(
            input_set=(query, key, value),
            param_set=(wq, wk, wv, wo),
            output_set=(result,))


class ViTMLP(Layer):
    def __init__(self,input,hidden,output,keep_prob=.5,
                 weight_init=initializer.Gaussian(),
                 bias_init=initializer.Constant()):
        layers = (
            Linear(input=input,output=hidden,weight_initializer=weight_init,
                   bias_initializer=bias_init),
            from_siso(gelu),
            from_siso(dropout, keep_prob),
            Linear(input=hidden,output=output,weight_initializer=weight_init,
                   bias_initializer=bias_init),
            from_siso(dropout, keep_prob),
        )
        net = Sequence(layers)

        super().__init__(
            input_set=net.input_set(),
            param_set=net.param_set(),
            output_set=net.output_set())


class ViTBlock(Layer):
    def __init__(self, num_token, dim_token, num_heads, mlp_num_hiddens, 
                 keep_prob=1, 
                 attention_weight_init=initializer.Gaussian(),
                 linear_weight_init=initializer.Gaussian(),
                 linear_bias_init=initializer.Constant(),
                 ln_gain_init=initializer.Constant(1.0),
                 ln_bias_init=initializer.Constant()):
        assert dim_token % num_heads == 0

        x = Variable()
        gain1 = Variable(shape=(1,num_token,dim_token),
                         initializer=ln_gain_init, name='ln1.gain')
        bias1 = Variable(shape=(1,num_token,dim_token),
                         initializer=ln_bias_init, name='ln1.bias')
        gain2 = Variable(shape=(1,num_token,dim_token),
                         initializer=ln_gain_init, name='ln2.gain')
        bias2 = Variable(shape=(1,num_token,dim_token),
                         initializer=ln_bias_init, name='ln2.bias')

        # 第一层
        tmp = plus(dot(layer_norm(x), gain1), bias1)
        inputs, params1, outputs = setup(DotProdMultiHeadAttention(
            num_heads=num_heads, p_o=dim_token, p_qk=dim_token//num_heads,
            p_v=dim_token//num_heads,n_q=num_token,n_kv=num_token,
            d_qk=dim_token,d_v=dim_token,keep_prob=keep_prob,
            kernel_init=attention_weight_init))
        link_pairs(tmp, inputs[0]); link_pairs(tmp, inputs[1]); 
        link_pairs(tmp, inputs[2])
        y = plus(outputs[0], x)

        # 第二层
        tmp = plus(dot(layer_norm(y), gain2), bias2)
        inputs, params2, outputs = setup(ViTMLP(
            input=dim_token,hidden=mlp_num_hiddens,output=dim_token,
            keep_prob=keep_prob, weight_init=linear_weight_init,
            bias_init=linear_bias_init))
        link_pairs(tmp, inputs[0])
        result = plus(y, outputs[0])

        super().__init__(
            input_set=(x,),
            param_set=(gain1,bias1) + params1 + (gain2,bias2) + params2,
            output_set=(result,),)
