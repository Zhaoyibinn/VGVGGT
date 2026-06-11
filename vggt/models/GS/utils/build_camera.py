import numpy as np
import torch
from vggt.models.GS.scene.cameras import Camera
def build_gs_camera(K, ext, height,width,data_device="cuda"):
    """Construct a GS camera that matches PyBullet's OpenGL renderer."""
    B,N = ext.shape[:2]
    K = np.asarray(K.cpu().detach(), dtype=np.float32).reshape(B,N,3, 3)
    width = int(width)
    height = int(height)
    fx = K[:,:,0,0]
    fy = K[:,:,1,1]
    # if fx <= 0 or fy <= 0:
    #     raise ValueError("Camera intrinsics must have positive focal lengths.")

    FoVx = 2.0 * np.arctan(width / (2.0 * fx))
    FoVy = 2.0 * np.arctan(height / (2.0 * fy))

    ext_c2w = torch.linalg.inv(ext.reshape(-1, 4, 4)).reshape_as(ext)
    rotm = ext_c2w[:,:,:3, :3].cpu().detach().numpy()  # c2w
    position = ext_c2w[:,:,:3, 3].cpu().detach().numpy()


    # rotm = qvec2rotmat(q_wxyz)

    # Build an identity matrix for every camera in the batch/sequence
    eye4 = np.eye(4, dtype=np.float32)[None, None, :, :]
    T_c2w = np.tile(eye4, (B, N, 1, 1))
    T_c2w[..., 0:3, 0:3] = rotm
    T_c2w[..., 0:3, 3] = position

    T_w2c = np.linalg.inv(T_c2w)

    # # Only feed the first camera into Camera for now (legacy API expects a single matrix)
    # T_c2w = T_c2w.reshape(-1, 4, 4)[0]
    # T_w2c = T_w2c.reshape(-1, 4, 4)[0]


    image = torch.zeros((3, height, width), dtype=torch.float32)

    cam_list_all = []
    for i in range(B):
        cam_list_batch = []
        for j in range(N):
            cam = Camera(
                R=T_c2w[i,j,:3,:3],
                T=T_w2c[i,j,:3,3],
                FoVx=FoVx[i,j],
                FoVy=FoVy[i,j],
                image=image,
                gt_alpha_mask=None,
                data_device=data_device,
            )
            cam_list_batch.append(cam)
        cam_list_all.append(cam_list_batch)

    return cam_list_all