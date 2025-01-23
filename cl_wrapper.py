import pyopencl as cl
import os
import torch
import numpy as np

class OpenCLContext:
    ctx: cl._cl.Context
    queue: any
    prog: cl.Program

    def __init__(self, type: str = "cl", device_id: int = 0):

        self.ctx = cl.create_some_context(False, [device_id])
        self.queue = cl.CommandQueue(self.ctx)
        print(f"[ComfyUI-Benderesque-Nodes] Initialized OpenCL context for device {self.ctx.devices[0].name}")

        cl_code_file = open(os.path.dirname(os.path.realpath(__file__)) + "/node_functions.cl")
        cl_code = cl_code_file.read()
        cl_code_file.close()

        self.prog = cl.Program(self.ctx, cl_code).build()
        print(f"[ComfyUI-Benderesque-Nodes] Built OpenCL code")

global_ish_context = OpenCLContext()


def transform(
    te: torch.FloatTensor,
    newsize: tuple[int, int],
    locrotscale: torch.FloatTensor,
    interpolation: int,
    extension: int
):
    """
    Image is Batch x Width x Height x 4
    LocRotScale is Batch x Width x Height x 5
    Loc xy, Rot in rad, Scale xy
    ##### Interpolation: 
    - 0 = closest
    - 1 = linear
    - 2 = cubic
    ##### Extension:
    - 0 = clip
    - 1 = repeat
    - 2 = extend
    - 3 = mirror\n
    """
    ctx = global_ish_context
    mf = cl.mem_flags

    results = []

    for b in range(te.size()[0]):
        te_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te[b].numpy())
        locrotscale_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=locrotscale[b].numpy())
        
        pixels = newsize[0] * newsize[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 4, dtype=np.float32)

        tr = ctx.prog.transform
        tr( ctx.queue, (pixels,), None, te_buf, np.int32(te.size()[1]), np.int32(te.size()[2]), np.int32(newsize[0]), np.int32(newsize[1]),
            locrotscale_buf, np.int32(interpolation), np.int32(extension), res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        results.append(torch.from_numpy(res_np_floats).to(dtype=torch.float32).reshape((te.size()[1], te.size()[2], 4)))

    return torch.stack(results, dim=0)