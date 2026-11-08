import os
import gc

os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
os.environ.setdefault('SPARSE_CONV_BACKEND', 'flex_gemm')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('FLEX_GEMM_ALGO', 'explicit_gemm')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault(
    'TRELLIS_DINOV3_PATH',
    os.path.expanduser('~/.cache/modelscope/hub/models/facebook/dinov3-vitl16-pretrain-lvd1689m'),
)

import cv2
import imageio
from PIL import Image
import torch

import xformers.ops.fmha
from trellis2.modules.sparse.conv import config as sparse_conv_config

# 避免 xformers 0.0.30 选择与当前 RTX 5090 不兼容的 FA3 内核
xformers.ops.fmha._set_use_fa3(False)

# 当前 TRELLIS.2 代码不会自动读取 FLEX_GEMM_ALGO 环境变量
sparse_conv_config.FLEX_GEMM_ALGO = 'explicit_gemm'

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("./assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp")
pipeline_type = os.environ.get('TRELLIS_PIPELINE_TYPE', '512')
print('pipeline_type:', pipeline_type)
print('FLEX_GEMM_ALGO:', os.environ.get('FLEX_GEMM_ALGO'))
mesh = pipeline.run(image, pipeline_type=pipeline_type)[0]
mesh.simplify(16777216) # nvdiffrast limit

# 4. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)
del video
gc.collect()
torch.cuda.empty_cache()

# 5. Export to GLB
if os.environ.get('TRELLIS_EXPORT_GLB', '1') == '1':
    glb_face_limit = int(os.environ.get('TRELLIS_GLB_FACE_LIMIT', '1000000'))
    if mesh.faces.shape[0] > glb_face_limit:
        print('simplifying mesh for GLB:', mesh.vertices.shape[0], 'vertices,', mesh.faces.shape[0], 'faces')
        mesh.simplify(glb_face_limit, verbose=True)
        print('after GLB simplify:', mesh.vertices.shape[0], 'vertices,', mesh.faces.shape[0], 'faces')
        torch.cuda.empty_cache()

    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   int(os.environ.get('TRELLIS_GLB_DECIMATION_TARGET', '500000')),
        texture_size        =   int(os.environ.get('TRELLIS_GLB_TEXTURE_SIZE', '2048')),
        remesh              =   os.environ.get('TRELLIS_GLB_REMESH', '0') == '1',
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True,
    )
    glb.export("sample.glb")
else:
    print('skip GLB export; set TRELLIS_EXPORT_GLB=1 to enable it')