# ComfyUI-Blenderesque-Nodes
Blender-like nodes for ComfyUI.<br>

Install:
```

```

Usage:<br>
Nodes will accept most inputs, akin to Blender's implicit conversion between vectors, floats, colors, and image types thereof.<br>
Nodes will output Blender-like data, and an image for general Comfy usage.<br>
Blender's compositor works by storing every image at its original size, and resizing it via cropping/padding with black+1.0a to match that of I believe the first input. These nodes do so as well. Blender's compositor lacks an explicit canvas resize node, which can be a problem!<br>
You should prefer to use the Blender data, to reduce loss in precision from excessive colorspace transforms, or avoid very small images inteded for a simple preview, among other things.<br>

Currently implemented nodes:<br>
<span style="color:LightGreen">Green with ✔</span> = Implemented<br>
<span style="color:IndianRed">Red with ❌</span> = Not implemented<br>
<span style="color:DimGrey">Dark grey with -</span> = Probably won't be implemented?<br>
<SPAN STYLE="color:GoldenRod">Yellow with O</span> = Partially implemented, useable<br>
Hopefully I'll figure out a way to do colorramps at some point.<br>
Blender lacks some compositor-applicable shader nodes, so this list will also include some shader nodes.<br>
Currently the list is incomplete.<br>

<details>
<summary>Meta Utility Nodes</summary>
<ul>
<li><span style="color:IndianRed">Resize Canvas ❌</span></li>
<li><span style="color:IndianRed">Extract Data ❌ (Get image, mask, canvas xy, float, etc.)</span></li>
</ul>
</details>

<details>
<summary>Input</summary>
<ul>
<li><span style="color:LightGreen">Value ✔</span></li>
<li><span style="color:LightGreen">RGB ✔</span></li>
<li><span style="color:IndianRed">Bokeh Image ❌</span></li>
Most other input nodes seem redundant or not applicable.
</ul>
</details>

<details>
<summary>Color</summary>
<ul>
<li><span style="color:LightGreen">Brightness/Contrast ✔</span></li>
<li><span style="color:LightGreen">Gamma ✔</span></li>
<li><span style="color:LightGreen">Hue/Saturation/Value ✔</span></li>
<li><span style="color:LightGreen">Invert Color ✔</span></li>
<li><span style="color:DimGrey">Light Falloff -</span></li>
<li><span style="color:IndianRed">Mix Color ❌ (see Mix converter)</span></li>
<li><span style="color:DimGrey">RGB Curves -</span></li>
<li><span style="color:DimGrey">Color Balance -</span></li>
<li><span style="color:DimGrey">Color Correction -</span></li>
<li><span style="color:DimGrey">Hue Correct -</span></li>
<li><span style="color:IndianRed">Exposure ❌</span></li>
<li><span style="color:IndianRed">Tonemap ❌</span></li>
<br>
<li><span style="color:IndianRed">Alpha Over ❌</span></li>
<li><span style="color:IndianRed">Z Combine ❌</span></li>
<li><span style="color:IndianRed">Alpha Convert ❌</span></li>
<li><span style="color:IndianRed">Convert Colorspace ❌</span></li>
<li><span style="color:LightGreen">Set Alpha ✔</span></li>
</ul>
</details>

<details>
<summary>Converter</summary>
<ul>
<li><span style="color:GoldenRod">Blackbody O (Missing rec709->linear, very minor color difference)</span></li>
<li><span style="color:LightGreen">Clamp ✔</span></li>
<li><span style="color:DimGrey">Color Ramp -</span></li>
<li><span style="color:GoldenRod">Combine Color O (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Combine XYZ ✔</span></li>
<li><span style="color:DimGrey">Float Curve -</span></li>
<li><span style="color:GoldenRod">Map Range O (Missing input hiding, interp methods untested)</span></li>
<li><span style="color:IndianRed">Math ❌</span></li>
<li><span style="color:IndianRed">Mix ❌</span></li>
<li><span style="color:LightGreen">RGB to BW ✔</span></li>
<li><span style="color:GoldenRod">Separate Color O (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Separate XYZ ✔</span></li>
<li><span style="color:IndianRed">Vector Math ❌</span></li>
<li><span style="color:IndianRed">Wavelength ❌</span></li>
</ul>
</details>

<details>
<summary>Transform</summary>
<ul>
<li><span style="color:IndianRed">Rotate ❌</span></li>
<li><span style="color:IndianRed">Scale ❌</span></li>
<li><span style="color:IndianRed">Transform ❌</span></li>
<li><span style="color:IndianRed">Translate ❌</span></li>
<li><span style="color:IndianRed">Corner Pin ❌</span></li>
<li><span style="color:IndianRed">Crop ❌</span></li>
<li><span style="color:IndianRed">Displace ❌</span></li>
<li><span style="color:IndianRed">Flip ❌</span></li>
<li><span style="color:IndianRed">Map UV ❌</span></li>
<li><span style="color:IndianRed">Lens Distortion ❌</span></li>
<li><span style="color:IndianRed">Movie Distortion ❌</span></li>
</ul>
</details>