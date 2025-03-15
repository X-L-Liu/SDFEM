import torch
import torch.nn as nn
import torch.nn.functional as F

def svd_rec(mat: torch.Tensor, index: list[int], isSaved: bool = True):
    # The following code is for reassembling after singular value decomposition
    # isSaved=True: Reassemble after retaining the index component
    # isSaved=False: Reassemble after removing the index component

    if not isinstance(index, list):
        index = [index]
    if not isSaved:
        index = [x for x in range(mat.shape[-1]) if x not in index]
    u, s, v = torch.svd(mat)
    s_ = torch.zeros_like(s)
    for n in range(len(index)):
        s_[..., index[n]] = s[..., index[n]]
    s__ = torch.zeros([s.shape[0], s.shape[1], s.shape[2], s.shape[2]]).to(mat.device)
    for k in range(s.shape[2]):
        s__[..., k, k] = s_[..., k]
    mat_ = torch.matmul(torch.matmul(u, s__), torch.swapaxes(v, -2, -1))
    return mat_


def svd_cat(mat: torch.Tensor, index: list[int], isSaved: bool = True):
    # The following code is for concatenation after singular value decomposition,
    # with the concatenation dimension being along the channel.
    # isSaved=True: Concatenate after retaining the index component.
    # isSaved=False: Concatenate after removing the index component.

    if not isinstance(index, list):
        index = [index]
    if not isSaved:
        index = [x for x in range(mat.shape[-1]) if x not in index]
    u, s, v = torch.svd(mat)
    mat_ = torch.zeros([mat.shape[0], mat.shape[1] * len(index), mat.shape[2], mat.shape[3]]).to(mat.device)
    for n in range(len(index)):
        img = torch.matmul(u[..., index[n]:index[n] + 1], torch.swapaxes(v, -2, -1)[..., index[n]:index[n] + 1, :])
        w = s[..., index[n]:index[n] + 1].unsqueeze(-1).repeat(1, 1, mat.shape[-2], mat.shape[-1])
        mat_[:, mat.shape[1] * n:mat.shape[1] * (n + 1), ...] = torch.mul(img, w)
    return mat_


class SDFEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDFEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.chan_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Mish(inplace=True),
        )

    def forward(self, img):
        with torch.no_grad():
            x = svd_cat(img, [], isSaved=False)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.chan_att(x1)
        out = x1 * x2
        return out
