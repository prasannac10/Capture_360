# models/spherical.py
import torch, torch.nn.functional as F

def euler_to_matrix(yaw, pitch, roll):
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    R = torch.zeros((yaw.shape[0], 3, 3), device=yaw.device)
    R[:,0,0] = cy*cr + sy*sp*sr
    R[:,0,1] = sr*cp
    R[:,0,2] = -sy*cr + cy*sp*sr
    R[:,1,0] = -cy*sr + sy*sp*cr
    R[:,1,1] = cr*cp
    R[:,1,2] = sr*sy + cy*sp*cr
    R[:,2,0] = sy*cp
    R[:,2,1] = -sp
    R[:,2,2] = cy*cp
    return R

# def spherical_project(feats, poses, B, N, out_hw=(256, 512)):
#     device = feats.device
#     C = feats.shape[1]
#     Hs, Ws = out_hw

#     pano = torch.zeros(B, C, Hs, Ws, device=device)
#     weight = torch.zeros(B, 1, Hs, Ws, device=device)

#     theta = torch.linspace(-torch.pi, torch.pi, Ws, device=device)
#     phi   = torch.linspace(-torch.pi/2, torch.pi/2, Hs, device=device)
#     phi, theta = torch.meshgrid(phi, theta, indexing="ij")

#     dirs = torch.stack([
#         torch.cos(phi)*torch.sin(theta),
#         torch.sin(phi),
#         torch.cos(phi)*torch.cos(theta)
#     ], dim=-1)

#     feats = feats.view(B, N, C, *feats.shape[-2:])

#     for i in range(N):
#         R = euler_to_matrix(*poses[:,i].T)
#         d = torch.einsum("hwc,bcj->bhwj", dirs, R)

#         u = torch.atan2(d[...,0], d[...,2]) / torch.pi
#         v = d[...,1] / (torch.pi/2)
#         grid = torch.stack([u, v], dim=-1)

#         sampled = F.grid_sample(feats[:,i], grid, align_corners=True)
#         pano += sampled
#         weight += (sampled.abs().sum(1,keepdim=True)>0).float()

#     return pano / (weight + 1e-6)


# spherical.py

def spherical_project(features, rotations, pano_h, pano_w):
    """
    features: [N, C]
    rotations: [N,3,3]
    returns: [C, pano_h, pano_w]
    """

    C = features.shape[1]
    canvas = torch.zeros(C, pano_h, pano_w, device=features.device)
    weight = torch.zeros(1, pano_h, pano_w, device=features.device)

    for i in range(features.shape[0]):
        R = rotations[i]

        # Simplified mapping: yaw -> x, pitch -> y
        yaw = torch.atan2(R[1,0], R[0,0])
        pitch = torch.asin(-R[2,0])

        x = ((yaw + torch.pi) / (2 * torch.pi) * pano_w).long().clamp(0, pano_w-1)
        y = ((pitch + torch.pi/2) / torch.pi * pano_h).long().clamp(0, pano_h-1)

        canvas[:, y, x] += features[i]
        weight[:, y, x] += 1.0

    return canvas / (weight + 1e-6)