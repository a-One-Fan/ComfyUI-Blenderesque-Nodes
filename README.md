# ComfyUI-Blenderesque-Nodes
Blender-like nodes for ComfyUI.<br>

<h1>Work In Progress<br>
Nodes might change and you will need to redo your workflow! If a node is broken, delete it and re-add it.</h1>

<img src=mary_combo.png style="width:50%;height:50%">

Install:
```
cd ./custom_nodes
git clone https://github.com/a-One-Fan/ComfyUI-Blenderesque-Nodes
pip install -r ./ComfyUI-Blendersque-Nodes/requirements.txt
```

Usage:<br><br>
Nodes will accept most inputs, akin to Blender's implicit conversion between vectors, floats, colors, and image types thereof.<br>
Nodes will output Blender-like data, and an image for general Comfy usage.<br>
Differently sized images are automatically cropped/padded like Blender's compositor.
You should prefer to use the Blender-like data output between these nodes, to reduce loss in precision from excessive colorspace transforms, or avoid very small images inteded for a simple preview, among other possible issues.<br>

<details>
<summary>Changelog</summary>
<br>
<ul>
<li>Fix broken node resizing</li>
<br><br>
<li>Improvements to UV input node</li>
<li>Bugfix (clamping saturation) for converting from HSV/HSL</li>
<br><br>
<li>Minor bugfix/improvement for casting floats to colors</li>
<br><br>
<li>Better Voronoi randomness, F2, and an approximation for edge distance</li>
<br><br>
<li>Initial Voronoi texture (F1 only, 2D only)</li>
<br><br>
<li>Initial Noise texture (FBM only)</li>
<br><br>
<li>Added Checker texture</li>
<br><br>
<li>Added colored outputs to most nodes</li>
<br><br>
<li>Fixed incorrect number widgets being dragged sometimes</li>
<li>Added Brick texture</li>
</ul>
</details>
<br>

Currently implemented nodes:<br><br>
<span style="color:LightGreen">Something with ✅</span> = Implemented<br>
<span style="color:IndianRed">Something with ❌</span> = Not implemented<br>
<span style="color:DimGrey">Something with -</span> = Probably won't be implemented?<br>
<SPAN STYLE="color:GoldenRod">Something with 🔵</span> = Partially implemented, useable<br>
Hopefully I'll figure out a way to do colorramps at some point.<br>
Blender lacks some compositor-applicable shader nodes, so this list will also include some shader nodes.<br>
Currently the list is incomplete.<br>
<br>
Hopefully in the future I'll look into integrating this with https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node for maximum convenience. I highly recommend that Blender addon!<br>
<details>
<summary>Meta/Bugs/TODO</summary>
<ul>
<li><span style="color:IndianRed">Integrate with https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node ❌</span></li>
<li><span style="color:GoldenRod">Dynamic inputs 🔵</span></li>
<li><span style="color:GoldenRod">Dynamic widgets 🔵</span></li>
<li><span style="color:LightGreen">Merged input sockets and default values ("widgets") ✅</span></li>
<li><span style="color:IndianRed">Merged output blender and image sockets ❌</span></li>
<li><span style="color:IndianRed">Low precision on image transforms, teethy edges ❌</span></li>
<li>Currently, Comfy does not support loading EXR images. Various vector passes (e.g. UV) need 32 bit data, and will look much worse with 8 bit data. -</span></li>
</ul>
</details>

<details>
<summary>New Comfy-Specific Nodes 1/2 and new functionality to old nodes</summary>
<li><span style="color:LightGreen">Input/UV ✅</span></li>
Creates a basic UV gradient for mapping other textures with.
<li><span style="color:IndianRed">Converter/Extract Data ❌</span></li>
Convert Blender Data to float, image, or get its canvas size.
<br>
<br>
Functionality:
<li><span style="color:LightGreen">Transform/Crop - Uncrop (extend) and auto rescale ✅</span></li>
<li><span style="color:IndianRed">Transform/Lens Distortion - Handle alpha ❌</span></li>
</details>

<details>
<summary>Input 2/3</summary>
<ul>
<li><span style="color:LightGreen">Value ✅</span></li>
<li><span style="color:LightGreen">RGB ✅</span></li>
<li><span style="color:IndianRed">Bokeh Image ❌</span></li>
Most other input nodes seem redundant or not applicable.
</ul>
</details>

<details>
<summary>Color 6/12</summary>
<ul>
<li><span style="color:LightGreen">Brightness/Contrast ✅</span></li>
<li><span style="color:LightGreen">Gamma ✅</span></li>
<li><span style="color:LightGreen">Hue/Saturation/Value ✅</span></li>
<li><span style="color:LightGreen">Invert Color ✅</span></li>
<li><span style="color:DimGrey">Light Falloff -</span></li>
<li><span style="color:GoldenRod">Mix Color 🔵 (see Mix converter)</span></li>
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
<li><span style="color:LightGreen">Set Alpha ✅</span></li>
</ul>
</details>

<details>
<summary>Converter 11/12</summary>
<ul>
<li><span style="color:GoldenRod">Blackbody 🔵 (Missing rec709->linear, very minor color difference)</span></li>
<li><span style="color:LightGreen">Clamp ✅</span></li>
<li><span style="color:DimGrey">Color Ramp -</span></li>
<li><span style="color:GoldenRod">Combine Color 🔵 (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Combine XYZ ✅</span></li>
<li><span style="color:DimGrey">Float Curve -</span></li>
<li><span style="color:LightGreen">Map Range ✅</span></li>
<li><span style="color:LightGreen">Math ✅</span></li>
<li><span style="color:GoldenRod">Mix 🔵 (no non-uniform vector factor, no use alpha)</span></li>
<li><span style="color:LightGreen">RGB to BW ✅</span></li>
<li><span style="color:GoldenRod">Separate Color 🔵 (No colorspace option for YUV/YCbCr)</span></li>
<li><span style="color:LightGreen">Separate XYZ ✅</span></li>
<li><span style="color:IndianRed">Vector Math ❌</span></li>
<li><span style="color:LightGreen">Wavelength ✅</span></li>
</ul>
</details>

<details>
<summary>Transform 7/11</summary>
<ul>
<li><span style="color:GoldenRod">Rotate 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Scale 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Transform 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:GoldenRod">Translate 🔵 (No bicubic interpolation)</span></li>
<li><span style="color:IndianRed">Corner Pin ❌</span></li>
<li><span style="color:LightGreen">Crop ✅</span></li>
<li><span style="color:IndianRed">Displace ❌</span></li>
<li><span style="color:IndianRed">Flip ❌</span></li>
<li><span style="color:IndianRed">Map UV ✅ (Bit depth note*)</span></li>
<li><span style="color:GoldenRod">Lens Distortion 🔵 (Border of image is handled a bit differently than Blender; alpha is ignored, like in Blender)</span></li>
<li><span style="color:IndianRed">Movie Distortion ❌</span></li>
</ul>
</details>

<details>
<summary>Texture 4/9</summary>
<ul>
<li><span style="color:LightGreen"> Brick Texture ✅</span></li>
<li><span style="color:IndianRed"> Checker Texture ✅</span></li>
<li><span style="color:IndianRed"> Gabor Texture ❌</span></li>
<li><span style="color:IndianRed"> Gradient Texture ❌</span></li>
<li><span style="color:IndianRed"> Magic Texture ❌</span></li>
<li><span style="color:IndianRed"> Noise Texture 🔵 (fBM only, no distortion)</span></li>
<li><span style="color:IndianRed"> Voronoi Texture 🔵 (F1, F2 & edge distance only, edge distance is approximate, 2D only)</span></li>
<li><span style="color:IndianRed"> Wave Texture ❌</span></li>
<li><span style="color:IndianRed"> White Noise Texture ❌</span></li>
</ul>
</details>

<details>
<summary>Filter 0/10</summary>
<ul>
<li><span style="color:IndianRed"> Whatever blurs I can implement ❌</span></li>
<li><span style="color:IndianRed"> Anti-Aliasing ❌</span></li>
<li><span style="color:IndianRed"> Despeckle ❌</span></li>
<li><span style="color:IndianRed"> Dilate/Erode ❌</span></li>
<li><span style="color:IndianRed"> Filter Filter :) ❌</span></li>
<li><span style="color:IndianRed"> Glare ❌</span></li>
<li><span style="color:IndianRed"> Kuwahara ❌</span></li>
<li><span style="color:IndianRed"> Pixelate ❌</span></li>
<li><span style="color:IndianRed"> Posterize ❌</span></li>
<li><span style="color:IndianRed"> Sun Beams ❌</span></li>
</ul>
</details>

Implemented nodes: 31/59