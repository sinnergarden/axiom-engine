"""Stable Feature coverage, immutable storage and bounded projection outside Core."""
from .store import VerifiedMatrixStore
from .storage import write_buffer, write_part
from importlib import import_module

_BUILD_EXPORTS = {'feature_definitions':'.identity', 'select_feature_plan':'.identity',
                 'FeatureLibrary':'.library', 'FeatureRequest':'.library', 'FeatureView':'.library'}

def __getattr__(name):
    if name not in _BUILD_EXPORTS: raise AttributeError(name)
    value=getattr(import_module(_BUILD_EXPORTS[name],__name__),name)
    globals()[name]=value
    return value

__all__ = ['VerifiedMatrixStore','write_buffer','write_part',*_BUILD_EXPORTS]
