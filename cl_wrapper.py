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

def np_buf_to_te(buf, size, channels=4):
    """
    Size: Height x Width\n
    [Channels * Width * Height] -> [H] x [W] x [Channels]
    """
    res = torch.from_numpy(buf).to(dtype=torch.float32).reshape((size[0], size[1], channels))
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
        uvw_maybe_batch = uvw[b] if uvw.size()[0] > 1 else uvw[0]
        uvw_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(uvw_maybe_batch))
        
        pixels = uvw.size()[2] * uvw.size()[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 4, dtype=np.float32)

        mapuvw = ctx.prog.map_uv
        mapuvw( ctx.queue, (pixels,), None, 
            te_buf, np.int32(te.size()[2]), np.int32(te.size()[1]), 
            uvw_buf, np.int32(uvw.size()[2]), np.int32(uvw.size()[1]),
            np.int32(interpolation), np.int32(extension), res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        results.append(np_buf_to_te(res_np_floats, (uvw.size()[1], uvw.size()[2])))

    return torch.stack(results, dim=0)

def brick_texture(
    offset: float,
    frequency_offset: int,
    squash: float,
    frequency_squash: int,
    uv: torch.FloatTensor,
    color1: torch.FloatTensor,
    color2: torch.FloatTensor,
    mortar: torch.FloatTensor,
    sssbwh: torch.FloatTensor,
):
    """
    UV is Batch x Height x Width x 2 \n
    color1, color2, mortar are Batch x Height x Width x 4\n
    sssbwh are Batch x Height x Width x 6, consisting of\n
    scale, mortar size, mortar smooth, bias, width and height\n
    \n
    Returns color and factor
    """
    ctx = global_ish_context
    mf = cl.mem_flags

    results_col = []
    results_fac = []

    assert uv.size()[3] == 2, f"UV is not 2-dimensional, is a {uv.size()} tensor instead"
    assert sssbwh.size()[3] == 6, f"SSSBWH is not 6-dimensional, is a {sssbwh.size()} tensor instead"

    for b in range(uv.size()[0]):
        uvw_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(uv[b]))
        color1_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(color1[b]))
        color2_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(color2[b]))
        mortar_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(mortar[b]))
        sssbwh_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(sssbwh[b]))
        
        pixels = uv.size()[2] * uv.size()[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 5 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 5, dtype=np.float32)

        cl_bricktex = ctx.prog.brick_texture
        cl_bricktex( ctx.queue, (pixels,), None, 
            uvw_buf, np.int32(uv.size()[2]), np.int32(uv.size()[1]), 
            color1_buf, color2_buf, 
            mortar_buf, sssbwh_buf,
            np.float32(offset), np.int32(frequency_offset), np.float32(squash), np.int32(frequency_squash),
            res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        res_te = np_buf_to_te(res_np_floats, (uv.size()[1], uv.size()[2]), 5)
        res_col, res_fac = res_te.split((4, 1), -1)

        results_col.append(res_col)
        results_fac.append(res_fac)

    return (torch.stack(results_col, dim=0), torch.stack(results_fac, dim=0))

def checker_texture(
    uvw: torch.FloatTensor,
    color1: torch.FloatTensor,
    color2: torch.FloatTensor,
    scale: torch.FloatTensor,
):
    """
    UVW is Batch x Height x Width x 3 \n
    color1, color2, are Batch x Height x Width x 4\n
    scale is Batch x Height x Width x 1\n
    Returns color and factor
    """
    ctx = global_ish_context
    mf = cl.mem_flags

    results_col = []
    results_fac = []

    assert uvw.size()[3] == 3, f"UVW is not 3-dimensional, is a {uvw.size()} tensor instead"
    assert scale.size()[3] == 1, f"Scale is not 1-dimensional, is a {scale.size()} tensor instead"

    for b in range(uvw.size()[0]):
        uvw_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(uvw[b]))
        color1_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(color1[b]))
        color2_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(color2[b]))
        scale_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(scale[b]))
        
        pixels = uvw.size()[2] * uvw.size()[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 5 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 5, dtype=np.float32)

        cl_checkertex = ctx.prog.checker_texture
        cl_checkertex( ctx.queue, (pixels,), None, 
            uvw_buf, np.int32(uvw.size()[2]), np.int32(uvw.size()[1]), 
            color1_buf, color2_buf, scale_buf,
            res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        res_te = np_buf_to_te(res_np_floats, (uvw.size()[1], uvw.size()[2]), 5)
        res_col, res_fac = res_te.split((4, 1), -1)

        results_col.append(res_col)
        results_fac.append(res_fac)

    return (torch.stack(results_col, dim=0), torch.stack(results_fac, dim=0))

def noise_texture(
    uv4: torch.FloatTensor,
    sdrlogd: torch.FloatTensor,
    noise_type: int,
    normalize: bool
):
    """
    UV4 is Batch x Height x Width x 4 \n
    sdrlogd is Batxh x Height x Width x 7, consisting of\n
    scale, detail, roughness, lacunarity, offset, gain, distortion\n
    noise_type is:\n
    0 - Multifractal\n
    1 - Ridged multifractal\n
    2 - Hybrid multifractal\n
    3 - Fractal Brownian Motion (fBM)\n
    4 - Hetero Terrain\n
    Returns color and factor
    """
    ctx = global_ish_context
    mf = cl.mem_flags

    results_col = []
    results_fac = []

    assert uv4.size()[3] == 4, f"UV4 is not 4-dimensional, is a {uv4.size()} tensor instead"
    assert sdrlogd.size()[3] == 7, f"sdrlogd is not 7-dimensional, is a {sdrlogd.size()} tensor instead"

    for b in range(uv4.size()[0]):
        uv4_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(uv4[b]))
        srdlogd_buf = cl.Buffer(ctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=te_to_np_buf(sdrlogd[b]))
        
        pixels = uv4.size()[2] * uv4.size()[1]
        res_cl_floats = cl.Buffer(ctx.ctx, mf.WRITE_ONLY, pixels * 3 * np.dtype(np.float32).itemsize)
        res_np_floats = np.empty(pixels * 3, dtype=np.float32)

        cl_noisetex = ctx.prog.noise_texture
        cl_noisetex( ctx.queue, (pixels,), None, 
            uv4_buf, np.int32(uv4.size()[2]), np.int32(uv4.size()[1]), 
            srdlogd_buf, np.int32(noise_type), np.int32(bool(normalize)),
            res_cl_floats)

        cl.enqueue_copy(ctx.queue, res_np_floats, res_cl_floats)

        res_te = np_buf_to_te(res_np_floats, (uv4.size()[1], uv4.size()[2]), 3)
        res_fac, _ = res_te.split((1, 2), -1)

        results_col.append(res_te)
        results_fac.append(res_fac)

    return (torch.stack(results_col, dim=0), torch.stack(results_fac, dim=0))