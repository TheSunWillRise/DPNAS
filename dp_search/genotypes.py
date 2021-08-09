from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_multi = namedtuple('Genotype', 'normal_bottom normal_concat_bottom reduce_bottom reduce_concat_bottom \
                                         normal_mid normal_concat_mid reduce_mid reduce_concat_mid \
                                         normal_top normal_concat_top')



PRIV_PRIMITIVES = [
    'none',
    'priv_max_pool_3x3',
    'priv_avg_pool_3x3',
    'priv_skip_connect',

    'priv_sep_conv_3x3_relu',
    'priv_sep_conv_3x3_elu',
    'priv_sep_conv_3x3_tanh',
    'priv_sep_conv_3x3_linear',
    'priv_sep_conv_3x3_htanh',
    'priv_sep_conv_3x3_sigmoid',
]



PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'resep_conv_3x3',
    'resep_conv_5x5',
]


