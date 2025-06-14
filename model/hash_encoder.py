import torch
import taichi as ti

from taichi.math import uvec3
from icecream import ic
from .utils import (
    data_type, 
    torch_type, 
    align_to,
    res_in_level_np,
    scale_in_level_np,
)


# 3D Hash Encoder
# =============================================================================
def build_hash_encoder_kernel(
    log_per_level_scale,
    base_res: float = 16.0,
    hash_level: int = 16,
    feat_dim: int = 2,
    begin_fast_hash_level: int = 16,
):
    """
    This function constructs a Taichi kernel that encodes
    3D coordinates into a hash map with multiple levels of resolution.

    Args:
    base_res (float, optional): Base resolution of the hash map. Default is 16.
    hash_level (int, optional): Number of levels in the hash map. Default is 16.
    feature_per_level (int, optional): Number of features per level. Default is 2.
    begin_fast_hash_level (int, optional): The level at which the fast hash method
    starts. Default is 16.

    Returns:
    A Taichi kernel, hash_encoder_kernel.
    """

    # Type
    feat_vec = ti.types.vector(
        n=feat_dim, 
        dtype=data_type,
    )

    # Functions
    @ti.func
    def fast_hash(pos_grid_local):
        result = ti.uint32(0)
        # tiny-cuda-nn may use different primes
        # primes = uvec3(ti.uint32(1), ti.uint32(1958374283), ti.uint32(2654435761))
        primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861))
        for i in ti.static(range(3)):
            result ^= ti.uint32(pos_grid_local[i]) * primes[i]
        return result

    @ti.func
    def under_hash(pos_grid_local, resolution):
        result = ti.uint32(0)
        stride = ti.uint32(1)
        for i in ti.static(range(3)):
            result += ti.uint32(pos_grid_local[i] * stride)
            stride *= resolution
        return result


    @ti.func
    def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
        hash_result = ti.uint32(0)
        if indicator:
            hash_result = under_hash(pos_grid_local, resolution)
        else:
            hash_result = fast_hash(pos_grid_local)

        return hash_result % map_size

    @ti.func
    def grid_scale(level, log_scale, base_res):
        exp_scale = ti.exp(level * log_scale)
        return base_res * exp_scale - 1.0

    @ti.func
    def grid_resolution(scale):
        return ti.uint32(ti.ceil(scale)) + 1

    if begin_fast_hash_level == hash_level:
        # if no fast_hash function required,
        # use a larger block_dim
        block_dim = 256
    else:
        block_dim = hash_level

    @ti.kernel
    def hash_encoder_kernel(
            xyzs: ti.types.ndarray(), 
            table: ti.types.ndarray(),
            output_embedding: ti.types.ndarray(), 
            hash_map_sizes: ti.types.ndarray(), 
            offsets: ti.types.ndarray(), 
            B: ti.i32,
        ):
        # get hash table embedding
        ti.loop_config(block_dim=block_dim)
        for i, level in ti.ndrange(B, hash_level):
            xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])

            scale = grid_scale(level, log_per_level_scale, base_res)
            resolution =  grid_resolution(scale)

            offset = offsets[level] * feat_dim

            pos = xyz * scale + 0.5
            pos_grid = ti.cast(ti.floor(pos), ti.uint32)
            pos -= ti.cast(pos_grid, data_type)

            map_size = hash_map_sizes[level]

            local_features = feat_vec(0.)

            for idx in ti.static(range(8)):
                w = 1.
                pos_grid_local = uvec3(0)

                for d in ti.static(range(3)):
                    if (idx & (1 << d)) == 0:
                        pos_grid_local[d] = pos_grid[d]
                        w *= 1 - pos[d]
                    else:
                        pos_grid_local[d] = pos_grid[d] + 1
                        w *= pos[d]

                index = grid_pos2hash_index(
                    level < begin_fast_hash_level,
                    pos_grid_local, 
                    resolution,
                    map_size,
                )
                index_table = ti.int32(
                    offset + index * feat_dim
                )

                for l_f in ti.static(range(feat_dim)):
                    local_features[l_f] += w * table[index_table+l_f]

            out_index_base = level * feat_dim 
            for l_f in ti.static(range(feat_dim)):
                output_embedding[i, out_index_base + l_f] = local_features[l_f]

    return hash_encoder_kernel

class HashEncoder_2D(torch.nn.Module):

    def __init__(
        self,
        max_params: float=2**19,
        levels: int=8,
        base_res: float=16,
        max_res: float=2048,
        feature_per_level: int=2,  
    ):
        super(HashEncoder_2D, self).__init__()

        # b=1.3195079565048218 fix value for 16 -> 1024
        self.log_b = scale_in_level_np(
            base_res=base_res,
            max_res=max_res,
            levels=levels,
        )
        # self.log_b = 1.587401032447815
        self.base_res = base_res
        self.hash_level = levels
        self.max_params = max_params
        self.feature_per_level = feature_per_level
        self.out_dim = feature_per_level * levels

        self.register_buffer(
            'offsets',
            torch.zeros(levels, dtype=torch.int32),
            persistent=False
        )
        self.register_buffer(
            'hash_map_sizes',
            torch.zeros(levels, dtype=torch.int32),
            persistent=False
        )

        offset = 0
        begin_fast_hash_level = levels
        for i in range(levels):
            resolution = res_in_level_np(
                i, base_res, self.log_b
            )
            full_size = resolution**3
            # Ensure that the parameter size is a multiple of 8.
            full_size_aligned = align_to(full_size, 8)

            # Restricted the parameter size using max_params.
            params_size_i = min(max_params, full_size_aligned)
            # print("resolution: ", resolution)

            self.offsets[i] = offset
            self.hash_map_sizes[i] = params_size_i

            # Record the first level that begins to use fast_hash
            if full_size > params_size_i:
                if begin_fast_hash_level == levels:
                    begin_fast_hash_level = i
            
            offset += params_size_i

        self.begin_fast_hash_level = begin_fast_hash_level
        self.total_param_size = offset * feature_per_level

        print(
            f'Hash Encoder: '
            f'base_res={base_res} '
            f'max_res={max_res} '
            f'hash_level={levels} '
            f'feat_per_level={feature_per_level} '
            f'per_level_scale={self.log_b} '
            f'total_hash_size={offset} '
        )

        self.hash_table = torch.nn.Parameter(
            torch.zeros(
                self.total_param_size,
                dtype=torch.float32,
            ),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.hash_table)

        self._hash_encoder_kernel = build_hash_encoder_kernel(
            self.log_b,
            base_res=self.base_res,
            hash_level=self.hash_level,
            feat_dim=self.feature_per_level,
            begin_fast_hash_level=self.begin_fast_hash_level,
        )

        # TODO: use a method to build the autograd function
        class _module_function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_pos, params):

                output_embedding = torch.empty(
                    input_pos.shape[0], self.out_dim,
                    dtype=torch_type,
                    device=input_pos.device, 
                    requires_grad=True,
                )
                self._hash_encoder_kernel(
                    input_pos,
                    params,
                    output_embedding,
                    self.hash_map_sizes,
                    self.offsets,
                    input_pos.shape[0],
                )
                ctx.save_for_backward(
                    input_pos, 
                    output_embedding, 
                    params
                )

                return output_embedding

            @staticmethod
            def backward(ctx, doutput):
                input_pos, output_embedding, params = ctx.saved_tensors
                output_embedding.grad = doutput

                self._hash_encoder_kernel.grad(
                    input_pos,
                    params,
                    output_embedding,
                    self.hash_map_sizes,
                    self.offsets,
                    input_pos.shape[0],
                )
                return None, params.grad

        self._module_function = _module_function.apply
        
    def forward(self, positions):
        return self._module_function(
            positions.contiguous(), 
            self.hash_table.contiguous(),
        )


# 2D Hash Encoder
# # =============================================================================
# def build_hash_encoder_kernel_2D(
#     log_per_level_scale,
#     base_res: float = 16.0,
#     hash_level: int = 16,
#     feat_dim: int = 2,
#     begin_fast_hash_level: int = 16,
# ):
#     """
#     This function constructs a Taichi kernel that encodes
#     3D coordinates into a hash map with multiple levels of resolution.

#     Args:
#     base_res (float, optional): Base resolution of the hash map. Default is 16.
#     hash_level (int, optional): Number of levels in the hash map. Default is 16.
#     feature_per_level (int, optional): Number of features per level. Default is 2.
#     begin_fast_hash_level (int, optional): The level at which the fast hash method
#     starts. Default is 16.

#     Returns:
#     A Taichi kernel, hash_encoder_kernel.
#     """

#     # Type
#     feat_vec = ti.types.vector(
#         n=feat_dim, 
#         dtype=data_type,
#     )

#     # Functions
#     @ti.func
#     def fast_hash(pos_grid_local):
#         result = ti.uint32(0)
#         # tiny-cuda-nn may use different primes
#         # primes = uvec3(ti.uint32(1), ti.uint32(1958374283), ti.uint32(2654435761))
#         primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861))
#         for i in ti.static(range(2)):
#             result ^= ti.uint32(pos_grid_local[i]) * primes[i]
#         return result

#     @ti.func
#     def under_hash(pos_grid_local, resolution):
#         result = ti.uint32(0)
#         stride = ti.uint32(1)
#         for i in ti.static(range(2)):
#             result += ti.uint32(pos_grid_local[i] * stride)
#             stride *= resolution
#         return result


#     @ti.func
#     def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
#         hash_result = ti.uint32(0)
#         if indicator:
#             hash_result = under_hash(pos_grid_local, resolution)
#         else:
#             hash_result = fast_hash(pos_grid_local)

#         return hash_result % map_size

#     @ti.func
#     def grid_scale(level, log_scale, base_res):
#         exp_scale = ti.exp(level * log_scale)
#         return base_res * exp_scale - 1.0

#     @ti.func
#     def grid_resolution(scale):
#         return ti.uint32(ti.ceil(scale)) + 1

#     if begin_fast_hash_level == hash_level:
#         # if no fast_hash function required,
#         # use a larger block_dim
#         block_dim = 256
#     else:
#         block_dim = hash_level

#     @ti.kernel
#     def hash_encoder_kernel(
#             xyzs: ti.types.ndarray(), 
#             table: ti.types.ndarray(),
#             output_embedding: ti.types.ndarray(), 
#             hash_map_sizes: ti.types.ndarray(), 
#             offsets: ti.types.ndarray(), 
#             B: ti.i32,
#         ):
#         # get hash table embedding
#         ti.loop_config(block_dim=block_dim)
#         for i, level in ti.ndrange(B, hash_level):
#             xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1]])

#             scale = grid_scale(level, log_per_level_scale, base_res)
#             resolution =  grid_resolution(scale)

#             offset = offsets[level] * feat_dim

#             pos = xyz * scale + 0.5
#             pos_grid = ti.cast(ti.floor(pos), ti.uint32)
#             pos -= ti.cast(pos_grid, data_type)

#             map_size = hash_map_sizes[level]

#             local_features = feat_vec(0.)

#             for idx in ti.static(range(4)):
#                 w = 1.
#                 pos_grid_local = uvec3(0)

#                 for d in ti.static(range(2)):
#                     if (idx & (1 << d)) == 0:
#                         pos_grid_local[d] = pos_grid[d]
#                         w *= 1 - pos[d]
#                     else:
#                         pos_grid_local[d] = pos_grid[d] + 1
#                         w *= pos[d]

#                 index = grid_pos2hash_index(
#                     level < begin_fast_hash_level,
#                     pos_grid_local, 
#                     resolution,
#                     map_size,
#                 )
#                 index_table = ti.int32(
#                     offset + index * feat_dim
#                 )

#                 for l_f in ti.static(range(feat_dim)):
#                     local_features[l_f] += w * table[index_table+l_f]

#             out_index_base = level * feat_dim 
#             for l_f in ti.static(range(feat_dim)):
#                 output_embedding[i, out_index_base + l_f] = local_features[l_f]

#     return hash_encoder_kernel
# class HashEncoder_2D(torch.nn.Module):

#     def __init__(
#         self,
#         max_params: float=2**10,
#         levels: int=8,
#         base_res: float=16,
#         max_res: float=512,
#         feature_per_level: int=2,  
#     ):
#         super(HashEncoder_2D, self).__init__()

#         # b=1.3195079565048218 fix value for 16 -> 1024
#         self.log_b = scale_in_level_np(
#             base_res=base_res,
#             max_res=max_res,
#             levels=levels,
#         )
#         # self.log_b = 1.587401032447815
#         self.base_res = base_res
#         self.hash_level = levels
#         self.max_params = max_params
#         self.feature_per_level = feature_per_level
#         self.out_dim = feature_per_level * levels

#         self.register_buffer(
#             'offsets',
#             torch.zeros(levels, dtype=torch.int32),
#             persistent=False
#         )
#         self.register_buffer(
#             'hash_map_sizes',
#             torch.zeros(levels, dtype=torch.int32),
#             persistent=False
#         )

#         offset = 0
#         begin_fast_hash_level = levels
#         for i in range(levels):
#             resolution = res_in_level_np(
#                 i, base_res, self.log_b
#             )
#             full_size = resolution**2
#             # Ensure that the parameter size is a multiple of 8.
#             full_size_aligned = align_to(full_size, 4)

#             # Restricted the parameter size using max_params.
#             params_size_i = min(max_params, full_size_aligned)
#             # print("resolution: ", resolution)

#             self.offsets[i] = offset
#             self.hash_map_sizes[i] = params_size_i

#             # Record the first level that begins to use fast_hash
#             if full_size > params_size_i:
#                 if begin_fast_hash_level == levels:
#                     begin_fast_hash_level = i
            
#             offset += params_size_i

#         self.begin_fast_hash_level = begin_fast_hash_level
#         self.total_param_size = offset * feature_per_level

#         print(
#             f'Hash Encoder: '
#             f'base_res={base_res} '
#             f'max_res={max_res} '
#             f'hash_level={levels} '
#             f'feat_per_level={feature_per_level} '
#             f'per_level_scale={self.log_b} '
#             f'total_hash_size={offset} '
#         )

#         self.hash_table = torch.nn.Parameter(
#             torch.zeros(
#                 self.total_param_size,
#                 dtype=torch.float32,
#             ),
#             requires_grad=True
#         )
#         torch.nn.init.uniform_(self.hash_table)

#         self._hash_encoder_kernel = build_hash_encoder_kernel_2D(
#             self.log_b,
#             base_res=self.base_res,
#             hash_level=self.hash_level,
#             feat_dim=self.feature_per_level,
#             begin_fast_hash_level=self.begin_fast_hash_level,
#         )

#         # TODO: use a method to build the autograd function
#         class _module_function(torch.autograd.Function):
#             @staticmethod
#             def forward(ctx, input_pos, params):

#                 output_embedding = torch.empty(
#                     input_pos.shape[0], self.out_dim,
#                     dtype=torch_type,
#                     device=input_pos.device, 
#                     requires_grad=True,
#                 )
#                 ic("start to hash")
#                 self._hash_encoder_kernel(
#                     input_pos,
#                     params,
#                     output_embedding,
#                     self.hash_map_sizes,
#                     self.offsets,
#                     input_pos.shape[0],
#                 )
#                 ic("end to hash")
#                 ctx.save_for_backward(
#                     input_pos, 
#                     output_embedding, 
#                     params
#                 )

#                 return output_embedding

#             @staticmethod
#             def backward(ctx, doutput):
#                 input_pos, output_embedding, params = ctx.saved_tensors
#                 output_embedding.grad = doutput

#                 self._hash_encoder_kernel.grad(
#                     input_pos,
#                     params,
#                     output_embedding,
#                     self.hash_map_sizes,
#                     self.offsets,
#                     input_pos.shape[0],
#                 )
#                 return None, params.grad

#         self._module_function = _module_function.apply
        
#     def forward(self, positions):
#         return self._module_function(
#             positions.contiguous(), 
#             self.hash_table.contiguous(),
#         )
    