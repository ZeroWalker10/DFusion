#!/o0usr/bin/env python
# coding=utf-8
model_names = ['resnet50', 'densenet121', 'wide_resnet50_2']
src_datasets = [('imagenet', '/datasets/adv/imagenet/new_full_adv_1k'),]
victim_model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2',]
transformer_victim_model_names = ['swin_t', 'swin_v2_t', 'vit_b_16', 'maxvit_t',]
victim_datasets = [('imagenet', '/datasets/adv/imagenet/new_full_adv_1k'),
                   ]
attack_book = './new_full_imagenet_attack_book_1k.json'
test_output_path = './workspace/'
mul_test_evaluation_file = 'evaluation/cmul_test_evaluation.csv'

baseline_attack_methods = {
    'DFusion': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'n': 20,
    },
}

limit_layer_config = {
    'imagenet': {
        'resnet50': {
            'DFusion': {
                '1.conv1': {'gamma': 0.3, 'block_ratio': 0.9},
             },
         },
        'densenet121': {
            'DFusion': {
                '1.features.conv0': {'gamma': 0.3, 'block_ratio': 0.9}, 
            }
         },
        'vgg19': {
            'DFusion': {
                '1.features.0': {'gamma': 0.3, 'block_ratio': 0.9}, 
             },
         },
        'inception_v3': {
            'DFusion': {
                '1.Conv2d_1a_3x3': {'gamma': 0.3, 'block_ratio': 0.9},
             },
        },
        'wide_resnet50_2': {
            'DFusion': {
                '1.conv1': {'gamma': 0.3, 'block_ratio': 0.9},
            },
         },
    },
    'caltech101': {
        'resnet50': {
            'DFusion': {
                '1.conv1': {'gamma': 0.3, 'block_ratio': 0.9},
             },
         },
        'densenet121': {
            'DFusion': {
                '1.features.conv0': {'gamma': 0.3, 'block_ratio': 0.9}, 
            }
         },
        'vgg19': {
            'DFusion': {
                '1.features.0': {'gamma': 0.3, 'block_ratio': 0.9}, 
             },
         },
        'inception_v3': {
            'DFusion': {
                '1.Conv2d_1a_3x3': {'gamma': 0.3, 'block_ratio': 0.9},
             },
        },
        'wide_resnet50_2': {
            'DFusion': {
                '1.conv1': {'gamma': 0.3, 'block_ratio': 0.9},
            },
         },
    },
}
