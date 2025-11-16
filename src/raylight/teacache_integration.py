"""
TeaCache integration functions for raylight.
This module contains the teacache forward functions adapted for use in raylight.
"""
import math
import torch
import comfy.ldm.common_dit
import comfy.model_management as mm

from torch import Tensor
from einops import repeat
from typing import Optional

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.wan.model import sinusoidal_embedding_1d


SUPPORTED_MODELS_COEFFICIENTS = {
    "flux": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    "flux-kontext": [-1.04655119e+03, 3.12563399e+02, -1.69500694e+01, 4.10995971e-01, 3.74537863e-02],
    "ltxv": [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
    "lumina_2": [-8.74643948e+02, 4.66059906e+02, -7.51559762e+01, 5.32836175e+00, -3.27258296e-02],
    "hunyuan_video": [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    "hidream_i1_full": [-3.13605009e+04, -7.12425503e+02, 4.91363285e+01, 8.26515490e+00, 1.08053901e-01],
    "hidream_i1_dev": [1.39997273, -4.30130469, 5.01534416, -2.20504164, 0.93942874],
    "hidream_i1_fast": [2.26509623, -6.88864563, 7.61123826, -3.10849353, 0.99927602],
    "wan2.1_t2v_1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
    "wan2.1_t2v_14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    "wan2.1_i2v_480p_14B": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
    "wan2.1_i2v_720p_14B": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
    "wan2.1_t2v_1.3B_ret_mode": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
    "wan2.1_t2v_14B_ret_mode": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
    "wan2.1_i2v_480p_14B_ret_mode": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
    "wan2.1_i2v_720p_14B_ret_mode": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
}


def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result


def teacache_wanmodel_forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        model_type = transformer_options.get("model_type")
        enable_teacache = transformer_options.get("enable_teacache", True)
        cache_device = transformer_options.get("cache_device")

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        modulated_inp = e0.to(cache_device) if "ret_mode" in model_type else e.to(cache_device)
        if not hasattr(self, 'teacache_state'):
            self.teacache_state = {
                0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None},
                1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None}
            }

        def update_cache_state(cache, modulated_inp):
            if cache['previous_modulated_input'] is not None:
                try:
                    cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                    if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                        cache['should_calc'] = False
                    else:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                except:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
            cache['previous_modulated_input'] = modulated_inp
            
        b = int(len(x) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

        if enable_teacache:
            should_calc = False
            for k in cond_or_uncond:
                should_calc = (should_calc or self.teacache_state[k]['should_calc'])
        else:
            should_calc = True

        # Check if USP is applied (forward_orig exists and is different from forward)
        # USP patches forward_orig, so if _teacache_original_forward_orig exists,
        # it means USP was applied before teacache
        has_usp = hasattr(self, '_teacache_original_forward_orig')
        
        if not should_calc:
            # Use cached residual
            if has_usp:
                # For USP, residual is in final output space (after head and unpatchify)
                # We need to reconstruct the output from cached residual
                # Since we can't easily reverse head and unpatchify, we'll call forward_orig
                # but this defeats the purpose of caching. For now, we'll use a simpler approach:
                # apply residual to blocks output, then do head and unpatchify
                # But this requires having blocks output, which we don't have in cache
                # So for USP, we'll just call forward_orig (caching disabled for USP for now)
                # TODO: Implement proper caching for USP
                return self._teacache_original_forward_orig(
                    x, t, context, clip_fea=clip_fea, freqs=freqs,
                    transformer_options=transformer_options, **kwargs
                )
            else:
                # For non-USP, residual is after blocks, before head/unpatchify
                for i, k in enumerate(cond_or_uncond):
                    x[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(x.device)
                
                # Apply head and unpatchify
                x = self.head(x, e)
                x = self.unpatchify(x, grid_sizes)
                return x
        else:
            # Calculate new forward pass
            ori_x = x.to(cache_device)
            
            if has_usp:
                # USP is applied, call USP's forward_orig
                # USP's forward_orig already includes head and unpatchify
                # For teacache with USP, we need to calculate residual differently
                # We'll call forward_orig to get the full output, then calculate residual
                # But since forward_orig includes everything, we store the full output difference
                x_final = self._teacache_original_forward_orig(
                    x, t, context, clip_fea=clip_fea, freqs=freqs, 
                    transformer_options=transformer_options, **kwargs
                )
                # For USP, forward_orig returns the final output (after head and unpatchify)
                # So we store the difference in the final output space
                # When using cache, we'll apply this difference directly
                for i, k in enumerate(cond_or_uncond):
                    # Calculate what the output would be without blocks (just head+unpatchify of input)
                    # This is an approximation - ideally we'd want blocks output, but that's complex with USP
                    ori_x_processed = self.head(self.unpatchify(ori_x, grid_sizes), e).to(cache_device) if hasattr(self, 'head') else ori_x
                    self.teacache_state[k]['previous_residual'] = (x_final.to(cache_device) - ori_x_processed)[i*b:(i+1)*b]
                return x_final
            else:
                # No USP, use standard forward logic
                for i, block in enumerate(self.blocks):
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args):
                            out = {}
                            out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                            return out
                        out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                        x = out["img"]
                    else:
                        x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                
                # Calculate residual after blocks, before head and unpatchify
                for i, k in enumerate(cond_or_uncond):
                    self.teacache_state[k]['previous_residual'] = (x.to(cache_device) - ori_x)[i*b:(i+1)*b]
                
                # head
                x = self.head(x, e)
                
                # unpatchify
                x = self.unpatchify(x, grid_sizes)
                return x

