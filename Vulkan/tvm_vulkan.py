import tvm
import time
from tvm.contrib import cc
from tvm.contrib import util
import numpy as np

def simple_device_save_module():
  def check_code_gen(device):
    n = 128 
    A = tvm.placeholder ((n ,n), name='A', dtype="int32")
    B = tvm.placeholder ((n, n), name='B', dtype="int32")
    C = tvm.compute (A.shape, lambda *i: A(*i) +  B(*i), name='C')
    s = tvm.create_schedule (C.op)

    bx, tx = s[C].split (C.op.axis[0], factor=64)
    s[C].bind (bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind (tx, tvm.thread_axis("threadIdx.x"))
    l = tvm.lower(s, [A, B, C], simple_mode=True)
    print (l)
    
#    module = tvm.build(s, [A, B, C], device, target_host="llvm")
#    path = "/home/khalikov/ml_models/kernel.vulkan"
#    module.imported_modules[0].save(path)
#    print(module.imported_modules[0].get_source())
#    print("done")
  check_code_gen("vulkan")

if __name__ == "__main__":
  simple_device_save_module()
