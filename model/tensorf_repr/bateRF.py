from icecream import ic
from . import batBase
from . import tensoRF
import torch.nn.functional as torch_F
import torch.nn.functional as F
import torch
class BAT_VMSplit(batBase.BatBase, tensoRF.TensorVMSplit):
    def convolute_line(self, kernel, line):
        # kernel.shape = 1, 1, kernel_size
        # line.shape = 1, n_components, grid_size, 1
        batch_size = line.shape[1]
        kernel_size = kernel.shape[-1]
        padding_size = (kernel_size // 2, kernel_size //2)

        line = line.squeeze(-1).view(batch_size, 1, -1)
        line = torch_F.pad(line, padding_size, mode="replicate")
        line = torch_F.conv1d(line, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1)
        line = line.view(1, batch_size, -1).unsqueeze(-1)
        return line

    def convolute_plane(self, kernel , plane, H, W):
        # kernel.shape = 1,1,kernel_size
        # plane.shape = 1, n_components, grid_size_H, grid_size_W
        batch_size = plane.shape[1]
        kernel_size = kernel.shape[-1]
        padding_size = (kernel_size // 2, kernel_size //2)
        # performs seperated 2d convolution on plane
        # convolve along dim -1
        plane = plane.reshape(batch_size, H, W)
        plane = torch_F.pad(plane, padding_size, mode="replicate")
        plane = torch_F.conv1d(plane, kernel.expand(H,1,-1), bias=None, stride=1, padding=0, dilation=1, groups=H)
        # transpose
        plane = plane.view(batch_size, H, W).permute(0, 2, 1)
        # convolve along dim -2
        plane = torch_F.pad(plane, padding_size, mode="replicate")
        plane = torch_F.conv1d(plane, kernel.expand(W,1,-1), bias=None, stride=1, padding=0, dilation=1, groups=W)
        # transpose back
        plane = plane.view(1, batch_size, W, H).permute(0,1,3,2).contiguous()
        return plane

    def compute_densityfeature(self, xyz_sampled, kernel=None, c2f_mode=None, interp_mode="bilinear"):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)


        for idx_plane in range(len(self.density_plane)):
            # activation ( ensure potive )
            if self.component_wise_feature2density:
                planes = self.feature2density(self.density_plane[idx_plane])
                lines = self.feature2density(self.density_line[idx_plane])
            elif self.plane_feature2density:
                planes = self.feature2density(self.density_plane[idx_plane])
                lines = self.density_line[idx_plane]
            elif self.abs_components:
                planes = torch.abs(self.density_plane[idx_plane])
                lines = torch.abs(self.density_line[idx_plane])
            else:
                planes = self.density_plane[idx_plane]
                lines = self.density_line[idx_plane]
            # separable component-wise convolution
            if c2f_mode != None :
                batch_size= self.density_n_comp[idx_plane]
                grid_size = self.gridSize[self.matMode[idx_plane]]
                if c2f_mode in ["uniform-gaussian","uniform-average"]:
                    kernel_1d = kernel.expand(1, 1, -1) # 1, 1, kernel_size
                elif c2f_mode == "diff":
                    raise Exception("Not supported c2f_mode = diff")
                    # since second dimension need to store another axis on 2D plane in seperated convolution, we have no extra dimension to store the component-wise kernel
                if self.convolve_positive_only:
                    planes_sharp = planes
                planes = self.convolute_plane(kernel_1d,planes , grid_size[0], grid_size[1])
                if self.convolve_plane_only == False:
                    lines = self.convolute_line(kernel_1d, lines)

            # component-wise sample splits (interpolation)
            plane_coef_point = F.grid_sample(planes, coordinate_plane[[idx_plane]], mode=interp_mode, align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(lines, coordinate_line[[idx_plane]], mode=interp_mode, align_corners=True).view(-1, *xyz_sampled.shape[:1])
            if self.convolve_positive_only and c2f_mode!=None:
                if self.ignore_negative_split:
                    plane_coef_point = plane_coef_point * (line_coef_point.detach() >= 0)
                else:
                    # use sharp version if line_coef sample < 0
                    plane_sharp_coef_point = F.grid_sample(planes_sharp, coordinate_plane[[idx_plane]], mode=interp_mode, align_corners=True).view(-1, *xyz_sampled.shape[:1])
                    plane_coef_point = plane_coef_point * (line_coef_point.detach() >= 0) + plane_sharp_coef_point * (line_coef_point.detach() < 0)

            # accumulte feature (outter product)
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled, kernel=None, c2f_mode=None, interp_mode="bilinear"):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):

            if c2f_mode != None :
                # performs group-wise convolution with group_size = 1
                batch_size= self.app_n_comp[idx_plane]
                grid_size = self.gridSize[self.matMode[idx_plane]]
                if c2f_mode in ["uniform-gaussian","uniform-average"]:
                    kernel_1d = kernel.expand(1, 1, -1) # self.color_n_comp, 1, kernel_size
                    #kernels_2d = (kernel[... , None] * kernel[:, :, None, :]) # self.color_n_comp, 1, kernel_size, kernel_size
                elif c2f_mode == "diff":
                    raise Exception("Not supported c2f_mode = diff")
                    # since second dimension need to store another axis on 2D plane in seperated convolution, we have no extra dimension to store the component-wise kernel
                app_plane_blur = self.convolute_plane(kernel_1d, self.app_plane[idx_plane], grid_size[0], grid_size[1])
                app_line_blur = self.convolute_line(kernel_1d, self.app_line[idx_plane])
            else:
                app_plane_blur = self.app_plane[idx_plane]
                app_line_blur = self.app_line[idx_plane]

            # accumulate app feature
            plane_coef_point.append(F.grid_sample(app_plane_blur, coordinate_plane[[idx_plane]], mode=interp_mode,
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(app_line_blur, coordinate_line[[idx_plane]], mode=interp_mode,
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

class BAT_VMSplit_MLPDensity(BAT_VMSplit):
    def TV_loss_density(self, reg):
        return torch.tensor(0)
    def compute_densityfeature(self, xyz_sampled, kernel=None, c2f_mode=None, interp_mode="bilinear"):
        app_features = self.compute_appfeature(xyz_sampled, kernel=kernel, c2f_mode=c2f_mode, interp_mode=interp_mode)
        valid_rgbs, sigma_feature = self.renderModule(xyz_sampled, None, app_features, self.view_pe_progress, self.fea_pe_progress)
        return sigma_features

class BAT_CP(batBase.BatBase, tensoRF.TensorCP):
    def compute_densityfeature(self, xyz_sampled, kernel=None, c2f_mode=None, interp_mode="bilinear"):
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))# 3, B, n_sample
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2) # 3, B*n_sample, 1, 2 (grid_sample grid)

        batch_size= self.density_n_comp[0] # all elements of n_comp should be same in CP mode

        density_line_blur = []
        for ind_line in range(3):
            if c2f_mode != None :
                # performs group-wise convolution with group_size = 1
                if c2f_mode in ["uniform-gaussian","uniform-average"]:
                    kernels_1d = kernel.expand(batch_size, 1, -1) # self.density_n_comp, 1, kernel_size

                ic(kernels_1d.dtype)
                density_line_blur.append(torch_F.conv1d(self.density_line[ind_line].squeeze(-1), kernels_1d, bias=None, stride=1, padding="same", dilation=1, groups=batch_size).unsqueeze(-1))
            else:
                # no kernel (c2f_mode = None)
                density_line_blur.append(self.density_line[ind_line])
        # accumulte feature
        line_coef_point = F.grid_sample(density_line_blur[0], coordinate_line[[0]], # 1, self.density_n_component[i] , B*n_sample, 1
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1]) # self.density_n_component[i]*n_sample, B
        line_coef_point = line_coef_point * F.grid_sample(density_line_blur[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(density_line_blur[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        return sigma_feature
    def compute_appfeature(self, xyz_sampled, kernel=None, c2f_mode=None, interp_mode="bilinear"):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        batch_size= self.app_n_comp[0] # all elements of n_comp should be same in CP mode

        app_line_blur = []
        for ind_line in range(3):
            if c2f_mode != None :
                # performs group-wise convolution with group_size = 1
                if c2f_mode in ["uniform-gaussian","uniform-average"]:
                    kernels_1d = kernel.expand(batch_size, 1, -1) # self.density_n_comp, 1, kernel_size
                ic(kernels_1d.dtype)
                app_line_blur.append(torch_F.conv1d(self.app_line[ind_line].squeeze(-1), kernels_1d, bias=None, stride=1, padding="same", dilation=1, groups=batch_size).unsqueeze(-1))
            else:
                # no kernel (c2f_mode = None)
                app_line_blur.append(self.app_line[ind_line])
        # accumulte feature

        line_coef_point = F.grid_sample(app_line_blur[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(app_line_blur[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(app_line_blur[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
