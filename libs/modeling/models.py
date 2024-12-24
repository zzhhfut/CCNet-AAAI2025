import os

CCNet_backbones = {}
def register_CCNet_backbone(name):
    def decorator(cls):
        CCNet_backbones[name] = cls
        return cls
    return decorator

MTGC_blocks = {}
def register_MTGC_block(name):
    def decorator(cls):
        MTGC_blocks[name] = cls
        return cls
    return decorator

CCNet_meta_archs = {}
def register_CCNet_meta_arch(name):
    def decorator(cls):
        CCNet_meta_archs[name] = cls
        return cls
    return decorator

def make_CCNet_backbone(name, **kwargs):
    CCNet_backbone = CCNet_backbones[name](**kwargs)
    return CCNet_backbone

def make_MTGC_block(name, **kwargs):
    MTGC_block = MTGC_blocks[name](**kwargs)
    return MTGC_block

def make_CCNet_meta_arch(name, **kwargs):
    CCNet_meta_arch = CCNet_meta_archs[name](**kwargs)
    return CCNet_meta_arch
