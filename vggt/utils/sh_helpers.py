# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import isqrt
import torch
from einops import einsum
torch.backends.cuda.preferred_linalg_library("magma")
# try:
from e3nn.o3 import matrix_to_angles, wigner_D
# except ImportError:
#     from depth_anything_3.utils.logger import logger
    # logger.warn("Dependency 'e3nn' not found. Required for rotating the camera space SH coeff")


def project_to_so3_strict(M: torch.Tensor) -> torch.Tensor:
    if M.shape[-2:] != (3, 3):
        raise ValueError("Input must be a batch of 3x3 matrices (i.e., shape [..., 3, 3]).")

    # 1. Compute SVD
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.mH

    # 2. Handle reflection case (det = -1)
    det_U = torch.det(U)
    det_V = torch.det(V)
    is_reflection = (det_U * det_V) < 0
    correction_sign = torch.where(
        is_reflection[..., None],
        torch.tensor([1, 1, -1.0], device=M.device, dtype=M.dtype),
        torch.tensor([1, 1, 1.0], device=M.device, dtype=M.dtype),
    )
    correction_matrix = torch.diag_embed(correction_sign)
    U_corrected = U @ correction_matrix
    R_so3_initial = U_corrected @ V.transpose(-2, -1)

    # 3. Explicitly ensure determinant is 1 (or extremely close)
    current_det = torch.det(R_so3_initial)
    det_correction_factor = torch.pow(current_det, -1 / 3)[..., None, None]
    R_so3_final = R_so3_initial * det_correction_factor

    return R_so3_final


def rotate_sh(
    sh_coefficients: torch.Tensor,  # "*#batch n"
    rotations: torch.Tensor,  # "*#batch 3 3"
) -> torch.Tensor:  # "*batch n"
    # https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2452412653
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape

    with torch.autocast(device_type=rotations.device.type, enabled=False):
        rotations_float32 = rotations.to(torch.float32)

        # switch axes: yzx -> xyz
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).unsqueeze(0).to(rotations_float32)
        # permuted_rotations = torch.linalg.inv(P) @ rotations_float32 @ P
        permuted_rotations = P.transpose(-2, -1) @ rotations_float32 @ P

        # ensure rotation has det == 1 in float32 type
        permuted_rotations_so3 = project_to_so3_strict(permuted_rotations)

        alpha, beta, gamma = matrix_to_angles(permuted_rotations_so3)
        result = []
        for degree in range(isqrt(n)):
            with torch.device(device):
                sh_rotations = wigner_D(degree, alpha, -beta, gamma).type(dtype)
            sh_rotated = einsum(
                sh_rotations,
                sh_coefficients[..., degree**2 : (degree + 1) ** 2],
                "... i j, ... j -> ... i",
            )
            result.append(sh_rotated)

    return torch.cat(result, dim=-1)


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5