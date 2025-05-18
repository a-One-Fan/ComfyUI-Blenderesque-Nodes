import pyopencl as cl
import os
import torch
import numpy as np

class OpenCLContext:
    ctx: cl._cl.Context
    queue: any
    prog: cl.Program 
    """
    Dimensions taken are Width, Height
    """

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

def te_to_np_buf(te: torch.Tensor):
    """
    ([1]x) [Height] x [Width] x [Channels] -> [Channels * Width * Height]
    """
    if len(te.size()) == 4:
        te = te[0]
    return te.contiguous().ravel().numpy()

def np_buf_to_te(buf, size):
    """
    Size: Height x Width\n
    [Channels * Width * Height] -> [H] x [W] x [Channels]
    """
    res = torch.from_numpy(buf).to(dtype=torch.float32).reshape((size[0], size[1], 4))
    return res

def __transform_4chan(
    te: torch.FloatTensor,
    newsize: tuple[int, int],
    locrotscale: torch.FloatTensor,
    interpolation: int,
    extension: int
):
    ctx = global_ish_context
    mf = cl.mem_flags

    results = []

    for b in range(te.size()[0]):
        te_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(te[b]))
        lrs_maybe_batch = locrotscale[b] if locrotscale.size()[0] > 1 else locrotscale[0]
        locrotscale_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(lrs_maybe_batch))
        
        pixels = newsize[0] * newsize[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 4, dtype=np.float32)

        tr = ctx.prog.transform
        tr( ctx.queue, (pixels,), None, te_buf, np.int32(te.size()[2]), np.int32(te.size()[1]), np.int32(newsize[1]), np.int32(newsize[0]),
            locrotscale_buf, np.int32(interpolation), np.int32(extension), res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        results.append(np_buf_to_te(res_np_floats, newsize))

    return torch.stack(results, dim=0)

def transform(
    te: torch.FloatTensor,
    newsize: tuple[int, int],
    locrotscale: torch.FloatTensor,
    interpolation: int,
    extension: int
):
    """
    Image is Batch x Height x Width x 4\n
    Newsize is Height x Width\n
    LocRotScale is (Batch or 1) x Height x Width x 5\n
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

    lrs_size = locrotscale.size()
    assert lrs_size[3] == 5, f"Locrotscale is not 5-channel ({lrs_size})"
    assert lrs_size[0] == 1 or lrs_size[0] == te.size()[0], f"Locrotscale has wrong batch size ({lrs_size} != {te.size()})"
    assert lrs_size[2] == newsize[1], f"Locrotscale doesn't match output image X ({lrs_size} != {newsize})"
    assert lrs_size[1] == newsize[0], f"Locrotscale doesn't match output image Y ({lrs_size} != {newsize})"

    splits = list(te.split(4, dim=3))
    lastpad = 0
    if splits[-1].size()[3] < 4:
        lastpad = 4 - splits[-1].size()[-1]
        splits[-1] = torch.cat((splits[-1], torch.zeros((*splits[-1].size()[:-1], lastpad))), dim=3)
    cropped_splits = [__transform_4chan(s, newsize, locrotscale, interpolation, extension) for s in splits]
    if lastpad:
        cropped_splits[-1], _ = cropped_splits[-1].split((4-lastpad, lastpad), dim=3)
    return torch.cat(cropped_splits, dim=3)

def lens_distortion(
        te: torch.FloatTensor,
        distortion: torch.FloatTensor,
        dispersion: torch.FloatTensor,
        projector: bool,
        jitter: bool,
        fit: bool
):
    """
    Image is Batch x Height x Width x 4\n
    distortion is Batch x Height x Width x 1\n
    dispersion is Batch x Height x Width x 1\n
    """

    ctx = global_ish_context
    mf = cl.mem_flags

    results = []

    jitterfit = 1 * jitter + 2 * fit
    size = (te.size()[1], te.size()[2])

    for b in range(te.size()[0]):
        te_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(te[b]))

        distortion_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(distortion))
        dispersion_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(dispersion))
        
        pixels = size[0] * size[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 4, dtype=np.float32)

        if(projector):
            prog = ctx.prog.execute_projector_distortion
        else:
            prog = ctx.prog.execute_screen_distortion

        prog( ctx.queue, (pixels,), None, te_buf, np.int32(size[1]), np.int32(size[0]), distortion_buf, dispersion_buf,
             np.int32(jitterfit), res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        results.append(np_buf_to_te(res_np_floats, size))

    return torch.stack(results, dim=0)

def map_uvw(
    te: torch.FloatTensor,
    uvw: torch.FloatTensor,
    interpolation: int,
    extension: int
):
    """
    Image is Batch x Height A x Width A x 4\n
    UVW is 1 or Batch x Height B x Width B x 3 \n
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
        te_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(te[b]))
        uv_maybe_batch = uvw[b] if uvw.size()[0] > 1 else uvw[0]
        uv_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(uv_maybe_batch))
        
        pixels = uvw.size()[2] * uvw.size()[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 4, dtype=np.float32)

        tr = ctx.prog.map_uv
        tr( ctx.queue, (pixels,), None, 
            te_buf, np.int32(te.size()[2]), np.int32(te.size()[1]), 
            uv_buf, np.int32(uvw.size()[2]), np.int32(uvw.size()[1]),
            np.int32(interpolation), np.int32(extension), res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        results.append(np_buf_to_te(res_np_floats, (uvw.size()[1], uvw.size()[2])))

    return torch.stack(results, dim=0)