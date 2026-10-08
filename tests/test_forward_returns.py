"""Whole-buffer arithmetic compared with the original scalar label contract."""
import math
import struct
import unittest

import numpy as np

from axiom_engine.core import ContractError, execute_forward_returns


def readonly(values,dtype='<f8'):
    array=np.asarray(values,dtype=dtype)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


class ForwardReturnsTests(unittest.TestCase):
    def test_exact_original_divide_then_subtract_and_endpoint_boundaries(self):
        tiny=np.nextafter(0.0,1.0);maximum=np.finfo('float64').max
        pairs=[(8.,7.),(1e16,1e16-2.),(tiny,tiny),(tiny,2*tiny),(maximum,tiny),
               (tiny,maximum),(0.,2.),(-0.,2.),(-1.,2.),(2.,0.),(2.,-1.),
               (math.inf,2.),(2.,math.inf),(math.nan,2.),(2.,math.nan),(2.,2.)]
        # Vary the exponent without introducing another approximate formula.
        pairs += [(math.ldexp(1.125,a),math.ldexp(1.375,b))
                  for a,b in [(-500,500),(500,-500),(-20,-19),(20,19),(0,0)]]
        start=readonly([a for a,b in pairs]);end=readonly([b for a,b in pairs])
        mask=readonly([True]*(len(pairs)-1)+[False],'?')
        result=execute_forward_returns(start,end,endpoint_validity=mask)
        expected=[];flags=[]
        for a,b,enabled in zip(start,end,mask):
            valid=bool(enabled and math.isfinite(a) and math.isfinite(b) and a>0 and b>0)
            value=float(b)/float(a)-1.0 if valid else 0.0
            valid=valid and math.isfinite(value)
            expected.append(value if valid else 0.0);flags.append(valid)
        self.assertEqual(result['values'].tobytes(),struct.pack('<'+str(len(expected))+'d',*expected))
        np.testing.assert_array_equal(result['validity'],flags)
        self.assertEqual(result['values'].dtype,np.dtype('<f8'))
        self.assertEqual(result['validity'].dtype,np.dtype('?'))
        self.assertTrue(np.isfinite(result['values']).all())

    def test_owned_readonly_outputs_and_empty_block(self):
        alias=np.array([8.,16.],dtype='<f8');start=alias.view();start.flags.writeable=False
        result=execute_forward_returns(start,readonly([7.,14.]),endpoint_validity=readonly([True,True],'?'))
        alias[:]=1.
        np.testing.assert_array_equal(result['values'],[-.125,-.125])
        for value in result.values():
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):value.flags.writeable=True
        empty=execute_forward_returns(readonly([]),readonly([]),endpoint_validity=readonly([],'?'))
        self.assertEqual(empty['values'].shape,(0,));self.assertEqual(empty['validity'].shape,(0,))

    def test_existing_packed_buffer_shape_dtype_endian_and_bool_domain(self):
        good=readonly([1.,2.]);flags=readonly([True,True],'?')
        cases=[([1.,2.],good,flags),(good.reshape(1,2),good,flags),
               (readonly([1.,2.],'<f4'),good,flags),(readonly([1.,2.],'>f8'),good,flags),
               (readonly([1.,2.,3.,4.])[::2],good,flags),
               (np.array([1.,2.]),good,flags),(good,readonly([1.]),flags),
               (good,good,readonly([1,1],'u1')),(good,good,memoryview(b'\x01\x02').cast('?'))]
        for start,end,mask in cases:
            with self.subTest(start_type=type(start),mask_type=type(mask)),self.assertRaises(ContractError):
                execute_forward_returns(start,end,endpoint_validity=mask)
