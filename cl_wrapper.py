import pyopencl as cl
import os

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