import torch
import torch.nn as nn

BLI_YUV_ITU_BT601 = "BLI_YUV_ITU_BT601"
BLI_YUV_ITU_BT709 = "BLI_YUV_ITU_BT709"
BLI_YCC_ITU_BT601 = "BLI_YCC_ITU_BT601"
BLI_YCC_ITU_BT709 = "BLI_YCC_ITU_BT709"
BLI_YCC_JFIF_0_255 = "BLI_YCC_JFIF_0_255"

def tmix(t1, t2, fac):
    if fac.dtype == torch.bool:
        fac = fac.to(torch.float32)
    return t1*(1.0-fac) + t2*fac

#blender/source/blender/blenlib/intern/math_color.cc

def rgb_to_hsv(rgb: torch.Tensor):
    r, g, b = rgb.split(1, dim=-1)

    k = (g < b).to(torch.float32) * -1.0
    newg = tmix(g, b, g < b) #swap g, b
    newb = tmix(b, g, g < b)
    g, b = newg, newb
    
    min_gb = b
    
    k = tmix(k, -2.0/6.0 - k, r < g)
    newr = tmix(r, g, r < g)
    newg = tmix(g, r, r < g)
    r, g, = newr, newg
    min_gb = torch.min(g, b)

    chroma = r - min_gb

    r_h = torch.abs(k + (g - b) / (6.0 * chroma + 1e-20))
    r_s = chroma / (r + 1e-20)
    r_v = r

    return torch.cat((r_h, r_s, r_v), dim=-1)

def hsv_to_rgb(hsv: torch.Tensor):
    h, s, v = hsv.split(1, dim=-1)

    nr = torch.abs(h * 6.0 - 3.0) - 1.0
    ng = 2.0 - torch.abs(h * 6.0 - 2.0)
    nb = 2.0 - torch.abs(h * 6.0 - 4.0)

    torch.clamp(nr, 0.0, 1.0)
    torch.clamp(ng, 0.0, 1.0)
    torch.clamp(nb, 0.0, 1.0)

    r_r = ((nr - 1.0) * s + 1.0) * v
    r_g = ((ng - 1.0) * s + 1.0) * v
    r_b = ((nb - 1.0) * s + 1.0) * v

    return torch.cat((r_r, r_g, r_b), dim=-1)

def rgb_to_hsl(rgb: torch.Tensor):
    r, g, b = rgb.split(1, dim=-1)
    cmax = torch.max(torch.max(r, g), b)
    cmin = torch.min(torch.min(r, g), b)
    l = torch.min((cmax + cmin) / 2.0, torch.ones_like(cmax))

    d_non0 = tmix(cmax - cmin, torch.ones_like(cmax), cmax == cmin)
    s = tmix(tmix(d_non0 / (2.0 - cmax - cmin), d_non0 / (cmax + cmin), l <= 0.5), 0.0, torch.isclose(cmax,cmin))

    h_r = tmix(torch.zeros_like(g), 
               (g - b) / d_non0 + tmix(torch.full_like(g, 0.0), torch.full_like(g, 6.0), g < b), 
               torch.isclose(cmax, r))
    h_g = tmix(torch.zeros_like(b),
               (b - r) / d_non0 + 2.0,
               torch.isclose(cmax, g))
    h_b = tmix(torch.zeros_like(r),
               (r - g) / d_non0 + 4.0,
               torch.isclose(cmax, b))
    h = h_r + h_g + h_b
    h /= 6.0

    return torch.cat((h, s, l), dim=-1)

def hsl_to_rgb(hsl: torch.Tensor):
    h, s, l = hsl.split(1, dim=-1)

    nr = torch.abs(h * 6.0 - 3.0) - 1.0
    ng = 2.0 - torch.abs(h * 6.0 - 2.0)
    nb = 2.0 - torch.abs(h * 6.0 - 4.0)

    torch.clamp(nr, 0.0, 1.0)
    torch.clamp(nb, 0.0, 1.0)
    torch.clamp(ng, 0.0, 1.0)

    chroma = (1.0 - torch.abs(2.0 * l - 1.0)) * s

    r_r = (nr - 0.5) * chroma + l
    r_g = (ng - 0.5) * chroma + l
    r_b = (nb - 0.5) * chroma + l

    return torch.cat((r_r, r_g, r_b), dim=-1)

def rgb_to_yuv(rgb: torch.Tensor, colorspace = BLI_YUV_ITU_BT709):
    r, g, b = rgb.split(1, dim=-1)

    if colorspace == BLI_YUV_ITU_BT601:
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
    elif colorspace == BLI_YUV_ITU_BT709:
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        u = -0.09991 * r - 0.33609 * g + 0.436 * b
        v = 0.615 * r - 0.55861 * g - 0.05639 * b
    else:
        raise Exception(f"Invalid colorspace {colorspace} when converting RGB to YUV!")

    return torch.cat((y, u, v), dim=-1)

def yuv_to_rgb(yuv: torch.Tensor, colorspace = BLI_YUV_ITU_BT709):
    y, u, v = yuv.split(1, dim=-1)
    if colorspace == BLI_YUV_ITU_BT601:
        r = y + 1.140 * v
        g = y - 0.394 * u - 0.581 * v
        b = y + 2.032 * u
    elif colorspace == BLI_YUV_ITU_BT709:
        r = y + 1.28033 * v
        g = y - 0.21482 * u - 0.38059 * v
        b = y + 2.12798 * u
    else:
        raise Exception(f"Invalid colorspace {colorspace} when converting YUV to RGB!")

    return torch.cat((r, g, b), dim=-1)

def rgb_to_ycc(rgb: torch.Tensor, colorspace = BLI_YCC_ITU_BT709):
    r, g, b = rgb.split(1, dim=-1)
    y = 128.0
    cr = 128.0
    cb = 128.0

    sr = 255.0 * r
    sg = 255.0 * g
    sb = 255.0 * b

    if colorspace == BLI_YCC_ITU_BT601:
        y = (0.257 * sr) + (0.504 * sg) + (0.098 * sb) + 16.0
        cb = (-0.148 * sr) - (0.291 * sg) + (0.439 * sb) + 128.0
        cr = (0.439 * sr) - (0.368 * sg) - (0.071 * sb) + 128.0
    elif colorspace == BLI_YCC_ITU_BT709:
        y = (0.183 * sr) + (0.614 * sg) + (0.062 * sb) + 16.0
        cb = (-0.101 * sr) - (0.338 * sg) + (0.439 * sb) + 128.0
        cr = (0.439 * sr) - (0.399 * sg) - (0.040 * sb) + 128.0
    elif colorspace == BLI_YCC_JFIF_0_255:
        y = (0.299 * sr) + (0.587 * sg) + (0.114 * sb)
        cb = (-0.16874 * sr) - (0.33126 * sg) + (0.5 * sb) + 128.0
        cr = (0.5 * sr) - (0.41869 * sg) - (0.08131 * sb) + 128.0
    else:
        raise Exception(f"Invalid colorspace {colorspace} when converting RGB to YCbCr!")

    return torch.cat((y, cb, cr), dim=-1)

def ycc_to_rgb(ycc: torch.Tensor, colorspace = BLI_YCC_ITU_BT709):
    y, cb, cr = ycc.split(1, dim=-1)
    r = 128.0
    g = 128.0
    b = 128.0

    if colorspace == BLI_YCC_ITU_BT601:
        r = 1.164 * (y - 16.0) + 1.596 * (cr - 128.0)
        g = 1.164 * (y - 16.0) - 0.813 * (cr - 128.0) - 0.392 * (cb - 128.0)
        b = 1.164 * (y - 16.0) + 2.017 * (cb - 128.0)
    elif colorspace == BLI_YCC_ITU_BT709:
        r = 1.164 * (y - 16.0) + 1.793 * (cr - 128.0)
        g = 1.164 * (y - 16.0) - 0.534 * (cr - 128.0) - 0.213 * (cb - 128.0)
        b = 1.164 * (y - 16.0) + 2.115 * (cb - 128.0)
    elif colorspace == BLI_YCC_JFIF_0_255:
        r = y + 1.402 * cr - 179.456
        g = y - 0.34414 * cb - 0.71414 * cr + 135.45984
        b = y + 1.772 * cb - 226.816
    else:
        raise Exception(f"Invalid colorspace {colorspace} when converting YCbCr to RGB!")
    
    r_r = r / 255.0
    r_g = g / 255.0
    r_b = b / 255.0

    return torch.cat((r_r, r_g, r_b), dim=-1)

def rgb_to_hsv_primitive(r, g, b):
    res = rgb_to_hsv(torch.Tensor((r, g, b)))
    return res[0], res[1], res[2]

def rgb_to_hsl_primitive(r, g, b):
    res = rgb_to_hsl(torch.Tensor((r, g, b)))
    return res[0], res[1], res[2]

def rgb_to_yuv_primitive(r, g, b, colorspace=BLI_YUV_ITU_BT709):
    res = rgb_to_yuv(torch.Tensor((r, g, b)), colorspace=colorspace)
    return res[0], res[1], res[2]

def rgb_to_ycc_primitive(r, g, b, colorspace=BLI_YCC_ITU_BT709):
    res = rgb_to_ycc(torch.Tensor((r, g, b)), colorspace=colorspace)
    return res[0], res[1], res[2]

def hsv_to_rgb_primitive(r, g, b):
    res = hsv_to_rgb(torch.Tensor((r, g, b)))
    return res[0], res[1], res[2]

def hsl_to_rgb_primitive(r, g, b):
    res = hsl_to_rgb(torch.Tensor((r, g, b)))
    return res[0], res[1], res[2]

def yuv_to_rgb_primitive(r, g, b, colorspace=BLI_YUV_ITU_BT709):
    res = yuv_to_rgb(torch.Tensor((r, g, b)), colorspace=colorspace)
    return res[0], res[1], res[2]

def ycc_to_rgb_primitive(r, g, b, colorspace=BLI_YCC_ITU_BT709):
    res = ycc_to_rgb(torch.Tensor((r, g, b)), colorspace=colorspace)
    return res[0], res[1], res[2]

blackbody_table_r = [[1.61919106e+03, -2.05010916e-03, 5.02995757e+00],
                            [2.48845471e+03, -1.11330907e-03, 3.22621544e+00],
                            [3.34143193e+03, -4.86551192e-04, 1.76486769e+00],
                            [4.09461742e+03, -1.27446582e-04, 7.25731635e-01],
                            [4.67028036e+03, 2.91258199e-05, 1.26703442e-01],
                            [4.59509185e+03, 2.87495649e-05, 1.50345020e-01],
                            [3.78717450e+03, 9.35907826e-06, 3.99075871e-01]]

blackbody_table_r_emb = nn.Embedding.from_pretrained(torch.Tensor(blackbody_table_r))

blackbody_table_g = [
    [-4.88999748e+02, 6.04330754e-04, -7.55807526e-02],
    [-7.55994277e+02, 3.16730098e-04, 4.78306139e-01],
    [-1.02363977e+03, 1.20223470e-04, 9.36662319e-01],
    [-1.26571316e+03, 4.87340896e-06, 1.27054498e+00],
    [-1.42529332e+03, -4.01150431e-05, 1.43972784e+00],
    [-1.17554822e+03, -2.16378048e-05, 1.30408023e+00],
    [-5.00799571e+02, -4.59832026e-06, 1.09098763e+00]]

blackbody_table_g_emb = nn.Embedding.from_pretrained(torch.Tensor(blackbody_table_g))

blackbody_table_b = [
    [5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03],
    [2.40430366e-11, 5.55021075e-08, -1.98503712e-04, 2.89312858e-02],
    [-1.40949732e-11, 1.89878968e-07, -3.56632824e-04, 9.10767778e-02],
    [-3.61460868e-11, 2.84822009e-07, -4.93211319e-04, 1.56723440e-01],
    [-1.97075738e-11, 1.75359352e-07, -2.50542825e-04, -2.22783266e-02],
    [-1.61997957e-13, -1.64216008e-08, 3.86216271e-04, -7.38077418e-01],
    [6.72650283e-13, -2.73078809e-08, 4.24098264e-04, -7.52335691e-01]]

blackbody_table_b_emb = nn.Embedding.from_pretrained(torch.Tensor(blackbody_table_b))

def blackbody_temperature_to_rec709(t):
    i = torch.zeros_like(t, dtype=torch.int)
    i += t >= 965.0
    i += t >= 1167.0
    i += t >= 1449.0
    i += t >= 1902.0
    i += t >= 3315.0
    i += t >= 6365.0

    permute_size = [i-1 for i in range(len(t.size())+1)]
    permute_size[0] = len(t.size())
    r = blackbody_table_r_emb(i).permute(permute_size)
    g = blackbody_table_g_emb(i).permute(permute_size)
    b = blackbody_table_b_emb(i).permute(permute_size)

    t_inv = 1.0 / t
    rec709_r = r[0] * t_inv + r[1] * t + r[2]
    rec709_g = g[0] * t_inv + g[1] * t + g[2]
    rec709_b = ((b[0] * t + b[1]) * t + b[2]) * t + b[3]

    rec709_r = tmix(rec709_r, 0.8262954810464208, t >= 12000.0)
    rec709_g = tmix(rec709_g, 0.9945080501520986, t >= 12000.0)
    rec709_b = tmix(rec709_b, 1.566307710274283, t >= 12000.0)

    rec709_r = tmix(rec709_r, 5.413294490189271, t < 800.0)
    rec709_g = tmix(rec709_g, -0.20319390035873933, t < 800.0)
    rec709_b = tmix(rec709_b, -0.0822535242887164, t < 800.0)

    return torch.cat((rec709_r, rec709_g, rec709_b), dim=-1)

