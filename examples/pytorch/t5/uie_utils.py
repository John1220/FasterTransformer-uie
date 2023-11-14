# -*- coding: utf-8 -*-


special_to_remove = {'<pad>', '</s>'}


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()
