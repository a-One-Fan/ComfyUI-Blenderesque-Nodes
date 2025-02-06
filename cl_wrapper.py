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

# ([1]x) [W] x [H] x [Channels] -> [Channels*Width*Height]
def te_to_np_buf(te: torch.Tensor):
    if len(te.size()) == 4:
        te = te[0]
    return te.permute((1, 0, 2)).contiguous().ravel().numpy()

# [Channels*Width*Height] -> [W] x [H] x [Channels]
def np_buf_to_te(buf, size):
    res = torch.from_numpy(buf).to(dtype=torch.float32).reshape((size[1], size[0], 4))
    return res.permute((1, 0, 2))

def transform(
    te: torch.FloatTensor,
    newsize: tuple[int, int],
    locrotscale: torch.FloatTensor,
    interpolation: int,
    extension: int
):
    """
    Image is Batch x Width x Height x 4\n
    LocRotScale is (Batch or 1) x Width x Height x 5\n
    Loc xy, Rot in rad, Scale xy\n
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

    lrs_size = locrotscale.size()
    assert lrs_size[3] == 5, f"Locrotscale is not 5-channel ({lrs_size})"
    assert lrs_size[0] == 1 or lrs_size[0] == te.size()[0], f"Locrotscale has wrong batch size ({lrs_size} != {te.size()})"
    assert lrs_size[1] == newsize[0], f"Locrotscale doesn't match output image X ({lrs_size} != {newsize})"
    assert lrs_size[2] == newsize[1], f"Locrotscale doesn't match output image Y ({lrs_size} != {newsize})"

    for b in range(te.size()[0]):
        te_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(te[b]))
        lrs_maybe_batch = locrotscale[b] if locrotscale.size()[0] > 1 else locrotscale[0]
        locrotscale_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(lrs_maybe_batch))
        
        pixels = newsize[0] * newsize[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 4, dtype=np.float32)

        tr = ctx.prog.transform
        tr( ctx.queue, (pixels,), None, te_buf, np.int32(te.size()[1]), np.int32(te.size()[2]), np.int32(newsize[0]), np.int32(newsize[1]),
            locrotscale_buf, np.int32(interpolation), np.int32(extension), res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        results.append(np_buf_to_te(res_np_floats, newsize))

    return torch.stack(results, dim=0)