"""Neutral, whole-buffer endpoint arithmetic; Research owns label qualification."""
import sys

from .contracts import ContractError, require
from .cs_batch import _snapshot_buffer


def execute_forward_returns(start, end, *, endpoint_validity):
    """Compute end/start - 1 once for a complete one-dimensional price block.

    Inputs use the existing packed Core ABI: readonly contiguous little-endian
    float64 endpoints and a same-length bool buffer, snapshotted before math.
    Missing/invalid endpoints, nonfinite or nonpositive prices, and nonfinite
    returns produce false validity and canonical +0.0. Positive subnormals have
    no epsilon threshold. Calendar, adjustment, PIT and maturity stay with the
    caller. This optional array entry requires axiom-engine[forward-returns].
    """
    require(sys.byteorder == 'little', 'BATCH_BUFFER_ENDIAN: little-endian host required')
    try:
        view=memoryview(start)
    except (TypeError,ValueError) as error:
        raise ContractError('BATCH_BUFFER: start') from error
    require(view.ndim==1,'BATCH_BUFFER_READONLY_1D_CONTIGUOUS: start')
    count=view.shape[0]
    opening,_=_snapshot_buffer(start,'values',count)
    closing,_=_snapshot_buffer(end,'values',count)
    validity,_=_snapshot_buffer(endpoint_validity,'value_validity',count)
    import numpy as np
    opening=np.frombuffer(opening,dtype='<f8');closing=np.frombuffer(closing,dtype='<f8')
    flags=np.frombuffer(validity,dtype='?').copy()
    flags &= np.isfinite(opening) & np.isfinite(closing) & (opening>0) & (closing>0)
    values=np.zeros(count,dtype='<f8')
    # Preserve the original two IEEE operations; (end-start)/start differs.
    with np.errstate(all='ignore'):
        np.divide(closing,opening,out=values,where=flags)
        np.subtract(values,1.0,out=values,where=flags)
    flags &= np.isfinite(values)
    values[~flags]=0.0
    return {'values':np.frombuffer(values.tobytes(),dtype='<f8'),
            'validity':np.frombuffer(flags.tobytes(),dtype='?')}
