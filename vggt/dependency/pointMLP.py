
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat
from .pointnet2_ops import pointnet2_utils

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]	
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu', batch_norm=True):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(out_channels),
                self.act
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
                self.act
            )
    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu', batch_norm=True):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        if batch_norm:
            self.net1 = nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                        kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(int(channel * res_expansion)),
                self.act
            )
        else:
            self.net1 = nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                        kernel_size=kernel_size, groups=groups, bias=bias),
                self.act
            )
        if groups > 1:
            if batch_norm:
                self.net2 = nn.Sequential(
                    nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                            kernel_size=kernel_size, groups=groups, bias=bias),
                    nn.BatchNorm1d(channel),
                    self.act,
                    nn.Conv1d(in_channels=channel, out_channels=channel,
                            kernel_size=kernel_size, bias=bias),
                    nn.BatchNorm1d(channel),
                )
            else:
                self.net2 = nn.Sequential(
                    nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                            kernel_size=kernel_size, groups=groups, bias=bias),
                    self.act,
                    nn.Conv1d(in_channels=channel, out_channels=channel,
                            kernel_size=kernel_size, bias=bias),
                )
        else:
            if batch_norm:
                self.net2 = nn.Sequential(
                    nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                            kernel_size=kernel_size, bias=bias),
                    nn.BatchNorm1d(channel)
                )
            else:
                self.net2 = nn.Sequential(
                    nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                            kernel_size=kernel_size, bias=bias),
                )
    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True, batch_norm=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation, batch_norm=batch_norm)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation, batch_norm=batch_norm)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu', batch_norm=True):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation, batch_norm=batch_norm)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True,
                 activation='relu', batch_norm=True, query_chunk_size=None):
        super(PointNetFeaturePropagation, self).__init__()
        self.query_chunk_size = query_chunk_size
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias, batch_norm=batch_norm)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation, batch_norm=batch_norm)

    def _interpolate_points(self, xyz1, xyz2, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            return points2.repeat(1, N, 1)

        if self.query_chunk_size is None or self.query_chunk_size <= 0 or N <= self.query_chunk_size:
            dists = square_distance(xyz1, xyz2)
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False, sorted=False)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            return torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        interpolated_chunks = []
        for start_idx in range(0, N, self.query_chunk_size):
            end_idx = min(start_idx + self.query_chunk_size, N)
            xyz1_chunk = xyz1[:, start_idx:end_idx]
            dists = square_distance(xyz1_chunk, xyz2)
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False, sorted=False)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_chunk = torch.sum(
                index_points(points2, idx) * weight.view(B, end_idx - start_idx, 3, 1),
                dim=2,
            )
            interpolated_chunks.append(interpolated_chunk)
        return torch.cat(interpolated_chunks, dim=1)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        interpolated_points = self._interpolate_points(xyz1, xyz2, points2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points




class PointMLP(nn.Module):
    def __init__(self, num_classes=50,points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2],
                 gmp_dim=64,cls_dim=64, **kwargs):
        super(PointMLP, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        self.act = get_activation(activation)

        # class label mapping
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label):
        xyz = x.permute(0, 2, 1)
        x = torch.cat([x,norm_plt],dim=1)
        x = self.embedding(x)  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)

        # here is the decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)

        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        #here is the cls_token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class PointMLPEncoder(nn.Module):
    def __init__(self, num_classes=50,points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2, 2, 2, 2],
                 gmp_dim=64,cls_dim=64, feature_channel=3, batch_norm=True,
                 preserve_input_dim=False, stage_dims=None, fp_query_chunk_size=None,
                 fp_query_chunk_size_last=None, **kwargs):
        """
        PointMLP 的编码器版本，只输出点级特征，不输出分类结果。

        forward 输入约定:
            x: 点坐标，形状为 [B, 3, N]
            norm_plt: 点特征，形状为 [B, C, N]，其中 C=feature_channel

        forward 输出:
            点级特征，形状为 [B, D, N]。其中 D 通常等于 de_dims[-1] + gmp_dim。

        参数说明:
            num_classes: 兼容原始分割版 PointMLP 保留下来的参数。当前编码器
                forward 不使用它，也不会影响输出维度。
            points: 预期输入点数。初始化时用它结合 reducers 计算每个 stage
                保留多少个 anchor 点。
            embed_dim: 输入嵌入层的输出通道数。
                1. preserve_input_dim=False 时，输入会先从 3+feature_channel
                   投影到 embed_dim。
                2. preserve_input_dim=True 且 embed_dim=None 时，入口不做降维。
                3. preserve_input_dim=True 且 embed_dim=feature_channel 时，入口
                   也是恒等映射，不改特征维度。
                4. preserve_input_dim=True 且 embed_dim 为其它值时，只对特征
                   分支做一次投影，不再和 xyz 在入口直接拼接。
            groups: 残差 1D 卷积里的分组卷积组数，只影响 ConvBNReLURes1D，
                不等于点云分组数。
            res_expansion: 残差块内部隐藏通道的扩张倍率。
            activation: 激活函数名称，可选值由 get_activation 决定，例如 relu、
                gelu、silu、leakyrelu 等。
            bias: 各个 Conv1d 层是否使用 bias。
            use_xyz: 在 LocalGrouper 中是否把局部邻域坐标 grouped_xyz 拼接到
                局部点特征上。开启后局部几何信息会显式参与特征提取。
            normalize: LocalGrouper 的局部归一化方式。
                1. "center": 以局部邻域均值做归一化。
                2. "anchor": 以 anchor 点特征和坐标做归一化。
                3. None: 不做该归一化。
            dim_expansion: 原始 PointMLP 每个 encoder stage 的通道扩张倍率。
                这是一个长度等于 stage 数的列表，第 i 项对应第 i 个 encoder
                stage。
                当 stage_dims=None 时，第 i 个 stage 的输出通道为
                last_channel * dim_expansion[i]。
            pre_blocks: 每个 encoder stage 中 PreExtraction 模块包含多少个
                残差块。这是一个按 stage 指定的列表，第 i 项对应第 i 个
                encoder stage。
                它控制的是局部邻域特征在做 max pooling 之前，要经过多少层
                ConvBNReLURes1D 残差提取。
                数值越大，该 stage 的局部特征建模越深，计算和显存开销也越大。
            pos_blocks: 每个 encoder stage 中 PosExtraction 模块包含多少个
                残差块。这同样是一个按 stage 指定的列表，第 i 项对应第 i 个
                encoder stage。
                它控制的是局部聚合完成之后，anchor 点特征还要再经过多少层
                ConvBNReLURes1D 做进一步提炼。
                可以把它理解为“邻域汇聚后”的特征细化深度。
            k_neighbors: 每个 anchor 点在每个 stage 中采样多少个最近邻点。
                这是一个按 stage 指定的列表，第 i 项对应第 i 个 encoder
                stage。
                数值越大，每个局部区域看见的邻域越大，但计算量也越高。
            reducers: 每个 stage 的下采样倍率。假设当前点数为 P，那么该层
                anchor 点数大致会变成 P // reducers[i]。
                这也是一个按 stage 指定的列表，第 i 项对应第 i 个 encoder
                stage。
                它决定每一层下采样得多快，也决定后面 LocalGrouper 中 FPS
                会保留多少个 anchor 点。
            de_dims: decoder 每个 PointNetFeaturePropagation 模块的输出通道。
                这是一个按 decoder stage 指定的列表，第 i 项对应第 i 个
                decoder stage。
                它决定特征上采样回高分辨率点集时各层的宽度，也直接影响最终
                输出特征维度中的局部分支宽度。
            de_blocks: decoder 中每个 PointNetFeaturePropagation 模块内部
                PosExtraction 的残差块数量。
                这是一个按 decoder stage 指定的列表，第 i 项对应第 i 个
                decoder stage。
                它控制 decoder 每一层在特征传播之后的细化深度。
            gmp_dim: 多尺度全局最大池化分支的通道数。每一层 encoder 特征先
                被映射到 gmp_dim，再汇总为最终的 global context。
            cls_dim: 原始分类/分割模型里的类别 token 分支宽度。当前编码器
                中虽然还保留了对应模块定义，但 forward 不使用它。
            feature_channel: 输入点特征维度，不包含 xyz 三维坐标。如果你的
                输入是 xyz + 1024 维特征，这里就应该设为 1024。
            batch_norm: 是否在本编码器创建的大多数卷积块中启用 BatchNorm1d。
            preserve_input_dim:
                1. False: 保持原始 PointMLP 风格，先把 xyz 和特征拼接，再立刻
                   投影到 embed_dim。
                2. True: 不在入口把 xyz 和特征直接拼接。xyz 只用于几何分支
                   的分组和邻域建模，特征分支可先保留原始宽度，再在后续 stage
                   中逐步压缩。
            stage_dims: 显式指定每个 encoder stage 的输出通道数。
                1. 为 None 时，使用 dim_expansion 自动逐层扩张。
                2. 不为 None 时，优先使用 stage_dims，覆盖 dim_expansion 的
                   通道计算逻辑。
                3. 这个参数在高维输入场景下尤其有用，例如输入是 1024 维特征
                   时，可以避免第一层之后通道数继续无控制膨胀。
                     4. 它也是一个按 stage 指定的列表，第 i 项对应第 i 个
                         encoder stage 的输出通道。
            **kwargs: 为兼容现有工厂函数和旧调用方式保留的额外参数，目前
                这个类本身不会主动消费这些参数。

        额外说明:
            1. 这个类最终返回的是点级特征，不是类别 logits。
            2. de_dims 会在初始化过程中被原地插入一项 de_dims.insert(0, en_dims[0])，
               因此如果外部复用同一个列表对象，最好传入新的列表副本。
            3. preserve_input_dim=True 只是不在入口立刻压缩特征，不代表后续
               stage 不会降维；后续通道仍由 stage_dims 或 dim_expansion 控制。
                4. 所有 encoder 侧列表参数 dim_expansion、pre_blocks、pos_blocks、
                    k_neighbors、reducers、stage_dims 都是“第 i 项控制第 i 个 encoder
                    stage”；decoder 侧 de_dims、de_blocks 则是“第 i 项控制第 i 个
                    decoder stage”。
        """
        super(PointMLPEncoder, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.feature_channel = feature_channel
        self.preserve_input_dim = preserve_input_dim
        if self.preserve_input_dim:
            if embed_dim is None or embed_dim == feature_channel:
                self.embedding = nn.Identity()
                last_channel = feature_channel
            else:
                self.embedding = ConvBNReLU1D(feature_channel, embed_dim, bias=bias, activation=activation, batch_norm=batch_norm)
                last_channel = embed_dim
        else:
            self.embedding = ConvBNReLU1D(3 + feature_channel, embed_dim, bias=bias, activation=activation, batch_norm=batch_norm)
            last_channel = embed_dim
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        if stage_dims is not None:
            assert len(stage_dims) == len(pre_blocks), \
                "Please check stage number consistent for stage_dims and pre_blocks."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = stage_dims[i] if stage_dims is not None else last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz, batch_norm=batch_norm)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation, batch_norm=batch_norm)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            query_chunk_size = fp_query_chunk_size_last if i == len(en_dims) - 2 else fp_query_chunk_size
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation, batch_norm=batch_norm,
                                           query_chunk_size=query_chunk_size)
            )

        self.act = get_activation(activation)

        # class label mapping
        # self.cls_map = nn.Sequential(
        #     ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation, batch_norm=batch_norm),
        #     ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation, batch_norm=batch_norm)
        # )
        self.cls_map = None
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation, batch_norm=batch_norm))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation, batch_norm=batch_norm)

        # # classifier
        # self.classifier = nn.Sequential(
        #     nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(),
        #     nn.Conv1d(128, num_classes, 1, bias=bias)
        # )
        self.en_dims = en_dims

    def forward(self, x, norm_plt):
        xyz = x.permute(0, 2, 1)
        if self.preserve_input_dim:
            x = norm_plt
        else:
            x = torch.cat([x, norm_plt], dim=1)
        x = self.embedding(x)  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)

        # here is the decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)
        # print(x.shape)
        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]
        # print(global_context.shape)

        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1)
        return x

def pointMLP(num_classes=50, **kwargs) -> PointMLP:
    return PointMLP(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, **kwargs)

def pointMLPEncoderBase(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBase2(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[3, 3, 3, 3], pos_blocks=[3, 3, 3, 3],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBase3(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 4, 4, 2], pre_blocks=[4, 4, 4, 4], pos_blocks=[4, 4, 4, 4],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 512, 512, 512], de_blocks=[4,4,4,4],
                 gmp_dim=128,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBase4(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 4, 4, 2], pre_blocks=[4, 4, 4, 4], pos_blocks=[4, 4, 4, 4],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBase5(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 4, 4, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 512, 512, 512], de_blocks=[4,4,4,4],
                 gmp_dim=128,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBase6(feature_channel=3, num_classes=50, batch_norm=True, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 512, 512, 512], de_blocks=[4,4,4,4],
                 gmp_dim=128,cls_dim=64, feature_channel=feature_channel, batch_norm=batch_norm, **kwargs)

def pointMLPEncoderBase7(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 4, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 1024, 1024, 1024], de_blocks=[4,4,4,4],
                 gmp_dim=512,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBase8(feature_channel=3, num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 4, 4], pos_blocks=[2, 2, 4, 4],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 512, 512, 512], de_blocks=[6,6,4,4],
                 gmp_dim=128,cls_dim=64, feature_channel=feature_channel, **kwargs)

def pointMLPEncoderBasezyb1(feature_channel=3, num_classes=50, batch_norm=True, output_dim=2048,
                            fp_query_chunk_size_last=2048, **kwargs) -> PointMLPEncoder:


    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=None, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[1, 1, 1, 1], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 512, 512, output_dim-256], de_blocks=[4,4,4,4],
                 gmp_dim=256,cls_dim=64, feature_channel=feature_channel, batch_norm=batch_norm, 
                 preserve_input_dim = True, fp_query_chunk_size_last=fp_query_chunk_size_last, **kwargs)


def pointMLPEncoder(num_classes=50, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 4, 4, 2], pre_blocks=[4, 4, 4, 4], pos_blocks=[4, 4, 4, 4],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 512, 512, 512], de_blocks=[4, 4, 4, 4],
                 gmp_dim=128,cls_dim=64, **kwargs)

if __name__ == '__main__':
    data = torch.rand(2, 3, 2048).cuda()
    norm = torch.rand(2, 3, 2048).cuda()
    cls_label = torch.rand([1, 16]).cuda()
    print("===> testing modelD ...")
    model = pointMLPEncoder(50)
    model.cuda()
    out = model(data, norm)  # [2,2048,50]
    print(out.shape)
    input()