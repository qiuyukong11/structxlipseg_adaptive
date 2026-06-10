# import torch
# import cv2 # OpenCV, not used in the provided snippets but often present in vision projects
# from PIL import Image # Pillow for image manipulation, not directly used in snippets
# import numpy as np # NumPy for numerical operations

# # --- Positional Embedding Interpolation ---
# def interpolate_pos_embeddings(model, new_image_size):
#     """
#     Interpolates the positional embeddings of a vision model to a new image size.
#     This is useful when you want to train or fine-tune a model on images of a
#     different resolution than it was originally designed for.
#     """
#     vision_model = model.vision_model
#     patch_size = vision_model.config.patch_size
#     # Calculate the number of patches for the new image size, including the CLS token
#     num_patches = (new_image_size // patch_size) ** 2 + 1

#     # Extract current positional embeddings
#     pos_embeddings = vision_model.embeddings.position_embedding.weight
#     # Reshape for interpolation: 1xEmbeddingDimxNumPatchesOld
#     pos_embeddings = pos_embeddings.unsqueeze(0).permute(0, 2, 1)

#     # Interpolate to the new number of patches using nearest neighbor
#     pos_embeddings = torch.nn.functional.interpolate(
#         pos_embeddings, size=(num_patches), mode='nearest'
#     ).squeeze(0).permute(1, 0)  # Reshape back: NumPatchesNewxEmbeddingDim

#     pos_embeddings = pos_embeddings.contiguous() # Ensure the tensor is contiguous in memory
#     # Update the model's positional embedding weights with the new interpolated ones
#     vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings)

#     # Update or register the position_ids buffer
#     # position_ids are typically a tensor like [[0, 1, 2, ..., num_patches-1]]
#     new_position_ids = torch.arange(0, num_patches, device=pos_embeddings.device).unsqueeze(0)
#     if hasattr(vision_model.embeddings, 'position_ids'):
#         vision_model.embeddings.position_ids = new_position_ids
#     else:
#         # If position_ids buffer doesn't exist, register it
#         vision_model.embeddings.register_buffer('position_ids', new_position_ids)

# def interpolate_text_pos_embeddings(model, new_max_token):
#     """
#     Interpolates the positional embeddings of a text model to a new maximum token length.
#     Similar to the vision model, this allows handling sequences of different lengths.
#     """
#     text_model = model.text_model
#     pos_embeddings = text_model.embeddings.position_embedding.weight
#     pos_embeddings = pos_embeddings.unsqueeze(0).permute(0, 2, 1)

#     pos_embeddings = torch.nn.functional.interpolate(
#         pos_embeddings, size=(new_max_token), mode='nearest'
#     ).squeeze(0).permute(1, 0)

#     pos_embeddings = pos_embeddings.contiguous()
#     text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings)

#     new_position_ids = torch.arange(0, new_max_token, device=pos_embeddings.device).unsqueeze(0)
#     if hasattr(text_model.embeddings, 'position_ids'):
#         text_model.embeddings.position_ids = new_position_ids
#     else:
#         text_model.embeddings.register_buffer('position_ids', new_position_ids)

# def longclip_pos_embeddings(model, new_max_token):
#     """
#     Extends text model positional embeddings using a specific scheme ("LongCLIP" style).
#     This method doesn't just interpolate all embeddings but tries to preserve
#     some initial embeddings and then uses a combination of existing embeddings
#     (linear interpolation between neighbors) to fill the new extended space.
#     It also includes an extrapolation part at the end.
#     """
#     text_model = model.text_model
#     pos_embeddings_pre = text_model.embeddings.position_embedding.weight
#     length, dim = pos_embeddings_pre.shape # Original length and dimension

#     keep_len = 20 # Number of initial embeddings to keep as is
#     # Calculate the maximum new length achievable with the 4x expansion scheme
#     # Each original embedding (after keep_len) effectively expands to 4 new embeddings.
#     # new_length = keep_len + (length - keep_len -1)*4 + 1 (for the last original one) - but this is not what the code calculates
#     # The code's new_length implies a specific expansion logic:
#     effective_expandable_len = length - keep_len
#     # The loop for i in range(length-1-keep_len) iterates (length - 1 - keep_len) times.
#     # Each iteration creates 4 new embeddings.
#     # The first keep_len are copied. The last original embedding is handled separately for extrapolation.
#     # The calculation new_length = 4*length - 3*keep_len seems to be an upper bound or derived from the paper's formula.
#     # Let's trace:
#     # If new_max_token is 248, original CLIP is 77.
#     # new_length_calc = 4*77 - 3*20 = 308 - 60 = 248. This matches the example case.

#     if 4 * length - 3 * keep_len < new_max_token: # Check if the requested new_max_token is too large for this scheme
#         raise ValueError(f"new_max_token ({new_max_token}) is too large for original length {length} with this expansion scheme. Max possible: {4*length - 3*keep_len}")

#     pos_embeddings_new = torch.zeros([new_max_token, dim], dtype=pos_embeddings_pre.dtype, device=pos_embeddings_pre.device)

#     # 1. Copy the first 'keep_len' embeddings
#     for i in range(keep_len):
#         pos_embeddings_new[i] = pos_embeddings_pre[i]

#     # 2. Interpolate/expand the middle part
#     # It iterates up to length-1-keep_len, meaning it processes original embeddings
#     # from index keep_len up to length-2.
#     for i in range(length - 1 - keep_len):
#         # Original embeddings being combined: pos_embeddings_pre[i + keep_len] and pos_embeddings_pre[i + 1 + keep_len]
#         # New embedding indices: from keep_len + 4*i
#         pos_embeddings_new[keep_len + 4*i]     = pos_embeddings_pre[i + keep_len]
#         pos_embeddings_new[keep_len + 4*i + 1] = (3 * pos_embeddings_pre[i + keep_len] + 1 * pos_embeddings_pre[i + 1 + keep_len]) / 4
#         pos_embeddings_new[keep_len + 4*i + 2] = (2 * pos_embeddings_pre[i + keep_len] + 2 * pos_embeddings_pre[i + 1 + keep_len]) / 4
#         pos_embeddings_new[keep_len + 4*i + 3] = (1 * pos_embeddings_pre[i + keep_len] + 3 * pos_embeddings_pre[i + 1 + keep_len]) / 4

#     # 3. Extrapolate for the last few positions based on the last two original embeddings
#     # The index 4*length - 3*keep_len - 4 is the position corresponding to the original (length-1) embedding in the new scheme.
#     # It seems the original paper might have a slightly different indexing or this is an adaptation.
#     # This part fills the very end of the new_max_token sequence if new_max_token is exactly 4*length - 3*keep_len.
#     # It uses the last embedding pos_embeddings_pre[length-1] and its difference from pos_embeddings_pre[length-2] for extrapolation.
#     last_orig_emb_idx_in_new = keep_len + 4 * (length - 1 - keep_len) # Index for pos_embeddings_pre[length-1]
    
#     # This extrapolation part fills based on the last original embedding and the trend from the one before it.
#     # The indices used (e.g., 4*length -3*keep_len - 4) directly map to the end of the calculated maximum possible length.
#     # If new_max_token is smaller, some of these might not be used or might overwrite previous interpolations if indexing isn't careful.
#     # However, pos_embeddings_new is initialized to zeros, so if new_max_token is smaller, the tail remains zero or as per previous loop.
#     # This section seems to fill the final 4 slots corresponding to the expansion of the last original segment.
#     base_idx_extrapolate = 4 * (length - keep_len -1) + keep_len # this is where pos_embeddings_pre[length-1] effectively lands
    
#     # The code for extrapolation is:
#     # pos_embeddings_new[4*length -3*keep_len - 4] = pos_embeddings_pre[length-1] + 0*(delta)/4
#     # ... up to
#     # pos_embeddings_new[4*length -3*keep_len - 1] = pos_embeddings_pre[length-1] + 3*(delta)/4
#     # This places the original last embedding at index (max_new_len - 4), then extrapolates 3 more.
    
#     # Correctly, this should use the embedding at index (length-1) as the base for extrapolation
#     # The loop for interpolation goes up to (length-1-keep_len)-1 for `i`.
#     # So, `i+keep_len` goes up to `length-2`. `i+1+keep_len` goes up to `length-1`.
#     # This means the interpolation uses pairs up to (pos_embed[length-2], pos_embed[length-1]).
#     # The following extrapolation seems to be designed to fill the *very end* of the maximum possible expanded length.
    
#     # The last embedding from the loop (pos_embeddings_pre[length-1]) is placed at:
#     # keep_len + 4*(length-1-keep_len-1) + 3  if we consider the last part of the loop.
#     # Or, if we consider the extrapolation fills the space *after* all interpolations:
#     # The loop creates embeddings up to index: keep_len + 4*(length-1-keep_len) - 1
    
#     # The extrapolation part seems to be for the very end of the sequence if new_max_token allows.
#     # It uses the difference between the last two original embeddings as the extrapolation step.
#     delta_extrap = (pos_embeddings_pre[length - 1] - pos_embeddings_pre[length - 2])
#     # The actual indices for this extrapolation part depend on new_max_token.
#     # The provided indices like `4*length -3*keep_len - 1` refer to the maximum possible length this scheme can generate.
#     # If new_max_token is smaller than this, these specific lines might write out of bounds or to unused parts if not careful.
#     # Assuming new_max_token is the max possible (4*length - 3*keep_len):
#     idx_base_extrap = 4*length -3*keep_len - 4
#     if idx_base_extrap < new_max_token : pos_embeddings_new[idx_base_extrap]     = pos_embeddings_pre[length-1] + 0 * delta_extrap / 4
#     if idx_base_extrap+1 < new_max_token : pos_embeddings_new[idx_base_extrap + 1] = pos_embeddings_pre[length-1] + 1 * delta_extrap / 4
#     if idx_base_extrap+2 < new_max_token : pos_embeddings_new[idx_base_extrap + 2] = pos_embeddings_pre[length-1] + 2 * delta_extrap / 4
#     if idx_base_extrap+3 < new_max_token : pos_embeddings_new[idx_base_extrap + 3] = pos_embeddings_pre[length-1] + 3 * delta_extrap / 4

#     text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings_new)
#     new_position_ids = torch.arange(0, new_max_token, device=pos_embeddings_new.device).unsqueeze(0)
#     if hasattr(text_model.embeddings, 'position_ids'):
#         text_model.embeddings.position_ids = new_position_ids
#     else:
#         text_model.embeddings.register_buffer('position_ids', new_position_ids)

# # --- Pooling Functions ---
# def average_pool(last_hidden_states, attention_mask):
#     """Performs masked average pooling on the last hidden states."""
#     # Zero out tokens that are padded (via attention_mask)
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     # Sum the unmasked token embeddings and divide by the number of unmasked tokens
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# def last_token_pool(last_hidden_states, attention_mask):
#     """
#     Pools the hidden state of the last unmasked token.
#     Handles cases with potential left padding.
#     """
#     # Check if it's left-padded (all sequences have a token at the last position)
#     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#     if left_padding:
#         return last_hidden_states[:, -1] # Return the hidden state of the very last token
#     else:
#         # For right-padding, find the actual length of each sequence
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         # Gather the hidden states at the computed sequence lengths
#         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# # --- Distributed Training Utilities ---
# def batch_align(fabric, x):
#     """Gathers a tensor 'x' across all distributed processes using Lightning Fabric."""
#     x = fabric.all_gather(x, sync_grads=True) # Gathers and synchronizes gradients
#     # Reshape to combine the process dimension with the batch dimension
#     return x.view(x.shape[0] * x.shape[1], -1)

# # --- Loss Functions ---
# cls_criterion = torch.nn.CrossEntropyLoss()

# def clip_loss(logits):
#     """
#     Computes the symmetric contrastive loss for CLIP.
#     It calculates cross-entropy loss for image-to-text and text-to-image similarities.
#     """
#     # Ground truth: for a batch of N pairs, the i-th image corresponds to the i-th text
#     gt = torch.arange(len(logits), dtype=torch.long, device=logits.device)
#     # Loss for (image_features @ text_features.T)
#     loss_i2t = cls_criterion(logits, gt)
#     # Loss for (text_features @ image_features.T) by transposing logits
#     loss_t2i = cls_criterion(logits.t(), gt)
#     return (loss_i2t + loss_t2i) / 2.0

# # --- Model Utilities ---
# def print_trainable_parameters(fabric, model):
#     """Prints the number of trainable and total parameters in a model."""
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     fabric.print( # Use fabric.print for proper logging in distributed settings
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
#     )
#     # Note: torch.cuda.memory_allocated() shows memory used by tensors, not necessarily just the model.
#     # For a more accurate model memory footprint, one might sum param.element_size() * param.nelement().
#     fabric.print('Current CUDA memory allocated: {} bytes'.format(torch.cuda.memory_allocated(device=fabric.device)))

# utils/func.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional Embedding Interpolation (vision) ---
def interpolate_pos_embeddings(model, new_image_size: int):
    vision_model = model.vision_model
    patch_size = vision_model.config.patch_size
    num_patches = (new_image_size // patch_size) ** 2 + 1  # +1 for CLS

    pos_embeddings = vision_model.embeddings.position_embedding.weight  # [orig_len, dim]
    pos_embeddings = pos_embeddings.unsqueeze(0).permute(0, 2, 1)       # [1, dim, orig_len]

    pos_embeddings = F.interpolate(pos_embeddings, size=(num_patches), mode='nearest') \
                       .squeeze(0).permute(1, 0).contiguous()           # [num_patches, dim]

    vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings)

    new_position_ids = torch.arange(0, num_patches, device=pos_embeddings.device).unsqueeze(0)
    if hasattr(vision_model.embeddings, 'position_ids'):
        vision_model.embeddings.position_ids = new_position_ids
    else:
        vision_model.embeddings.register_buffer('position_ids', new_position_ids)

# --- Positional Embedding Interpolation (text) ---
def interpolate_text_pos_embeddings(model, new_max_token: int):
    text_model = model.text_model
    pos_embeddings = text_model.embeddings.position_embedding.weight  # [orig_len, dim]
    pos_embeddings = pos_embeddings.unsqueeze(0).permute(0, 2, 1)     # [1, dim, orig_len]

    pos_embeddings = F.interpolate(pos_embeddings, size=(new_max_token), mode='nearest') \
                       .squeeze(0).permute(1, 0).contiguous()         # [new_len, dim]

    # text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_embeddings)
    text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_new.detach())

    new_position_ids = torch.arange(0, new_max_token, device=pos_embeddings.device).unsqueeze(0)
    if hasattr(text_model.embeddings, 'position_ids'):
        text_model.embeddings.position_ids = new_position_ids
    else:
        text_model.embeddings.register_buffer('position_ids', new_position_ids)
def longclip_pos_embeddings_cocoop(model, new_max_token: int, keep_len: int = 20):
    text_model = model.text_model
    pos_embeddings_pre = text_model.embeddings.position_embedding.weight
    length, dim = pos_embeddings_pre.shape

    max_len = 8 * length - 7 * keep_len
    if new_max_token > max_len:
        raise ValueError(
            f"new_max_token ({new_max_token}) > max supported ({max_len}) for orig_len={length}, keep_len={keep_len}"
        )

    pos_new = torch.zeros([new_max_token, dim],
                          dtype=pos_embeddings_pre.dtype,
                          device=pos_embeddings_pre.device)

    # === 1. 前 keep_len ===
    upto = min(keep_len, new_max_token)
    pos_new[:upto] = pos_embeddings_pre[:upto]

    # === 2. 插值扩展 ===
    write_ptr = keep_len
    steps = max(0, length - 1 - keep_len)
    for i in range(steps):
        a = pos_embeddings_pre[keep_len + i]
        b = pos_embeddings_pre[keep_len + i + 1]
        for s in [
            a,
            (3 * a + 1 * b) / 4,
            (2 * a + 2 * b) / 4,
            (1 * a + 3 * b) / 4,
        ]:
            if write_ptr < new_max_token:
                pos_new[write_ptr] = s
                write_ptr += 1

    # === 3. 外推尾部 ===
    delta_extrap = (pos_embeddings_pre[-1] - pos_embeddings_pre[-2])
    for k in range(4):
        if write_ptr < new_max_token:
            pos_new[write_ptr] = pos_embeddings_pre[-1] + k * delta_extrap / 4
            write_ptr += 1
        else:
            break

    # === ✅ 关键更新部分 ===
    text_model.embeddings.position_embedding = nn.Embedding.from_pretrained(pos_new, freeze=False)
    new_position_ids = torch.arange(0, new_max_token, device=pos_new.device).unsqueeze(0)
    text_model.embeddings.position_ids = new_position_ids
    text_model.config.max_position_embeddings = new_max_token      # ✅ 同步 config
    model.config.text_config.max_position_embeddings = new_max_token  # ✅ 同步最上层 config
    model.text_model.embeddings.position_embedding.num_embeddings = new_max_token  # ✅ 确认 embedding 对齐

    print(f"[longclip] extended text position embeddings: {length} -> {new_max_token}")

# --- LongCLIP-style text positional extension ---
def longclip_pos_embeddings(model, new_max_token: int, keep_len: int = 20):
    """
    扩展 CLIP 文本位置向量到更长的 token 长度（默认兼容 248 等长度）。
    规则：保留前 keep_len 个；后续按 4x 展开（线性插值）；最后 4 个位置做外推。
    注意：最大可扩长度为 4*orig_len - 3*keep_len。
    """
    text_model = model.text_model
    pos_embeddings_pre = text_model.embeddings.position_embedding.weight  # [orig_len, dim]
    length, dim = pos_embeddings_pre.shape

    # max_len = 4 * length - 3 * keep_len
    max_len = 8 * length - 7 * keep_len
    if new_max_token > max_len:
        raise ValueError(
            f"new_max_token ({new_max_token}) > max supported ({max_len}) for orig_len={length}, keep_len={keep_len}"
        )

    pos_new = torch.zeros([new_max_token, dim],
                          dtype=pos_embeddings_pre.dtype,
                          device=pos_embeddings_pre.device)

    # 1) keep front
    upto = min(keep_len, new_max_token)
    if upto > 0:
        pos_new[:upto] = pos_embeddings_pre[:upto]

    # 2) 4x expand middle (from keep_len .. length-2, paired with next)
    # 写入位置从 keep_len 开始，每次写 4 个： [base, 3/1, 2/2, 1/3]
    write_ptr = keep_len
    steps = max(0, length - 1 - keep_len)  # pair count
    for i in range(steps):
        a = pos_embeddings_pre[keep_len + i]
        b = pos_embeddings_pre[keep_len + i + 1]
        slots = [
            a,
            (3 * a + 1 * b) / 4,
            (2 * a + 2 * b) / 4,
            (1 * a + 3 * b) / 4,
        ]
        for s_item in slots:
            if write_ptr < new_max_token:
                pos_new[write_ptr] = s_item
                write_ptr += 1

    # 3) extrapolate tail (最后 4 个，基于最后两原始嵌入的差)
    delta_extrap = (pos_embeddings_pre[length - 1] - pos_embeddings_pre[length - 2])
    tail = [
        pos_embeddings_pre[length - 1] + 0 * delta_extrap / 4,
        pos_embeddings_pre[length - 1] + 1 * delta_extrap / 4,
        pos_embeddings_pre[length - 1] + 2 * delta_extrap / 4,
        pos_embeddings_pre[length - 1] + 3 * delta_extrap / 4,
    ]
    for s_item in tail:
        if write_ptr < new_max_token:
            pos_new[write_ptr] = s_item
            write_ptr += 1
        else:
            break

    text_model.embeddings.position_embedding.weight = torch.nn.Parameter(pos_new)
    new_position_ids = torch.arange(0, new_max_token, device=pos_new.device).unsqueeze(0)
    if hasattr(text_model.embeddings, 'position_ids'):
        text_model.embeddings.position_ids = new_position_ids
    else:
        text_model.embeddings.register_buffer('position_ids', new_position_ids)


def extend_clip_text_context(clip_model, max_token_len: int) -> None:
    """Extend OpenAI CLIP text positions with the StructXLIP/LongCLIP interpolation rule."""
    old_pos = clip_model.positional_embedding
    old_len, dim = old_pos.shape
    if max_token_len <= old_len:
        return

    keep_len = 20
    max_supported = 4 * old_len - 3 * keep_len
    if max_token_len > max_supported:
        raise ValueError(
            f"max_token_len={max_token_len} exceeds LongCLIP-style limit {max_supported} "
            f"for original context length {old_len}"
        )

    new_pos = torch.zeros(max_token_len, dim, dtype=old_pos.dtype, device=old_pos.device)
    new_pos[:keep_len] = old_pos[:keep_len]
    for i in range(old_len - 1 - keep_len):
        src0 = old_pos[i + keep_len]
        src1 = old_pos[i + 1 + keep_len]
        dst = keep_len + 4 * i
        if dst < max_token_len:
            new_pos[dst] = src0
        if dst + 1 < max_token_len:
            new_pos[dst + 1] = (3 * src0 + src1) / 4
        if dst + 2 < max_token_len:
            new_pos[dst + 2] = (src0 + src1) / 2
        if dst + 3 < max_token_len:
            new_pos[dst + 3] = (src0 + 3 * src1) / 4

    delta = old_pos[-1] - old_pos[-2]
    idx = 4 * old_len - 3 * keep_len - 4
    for k in range(4):
        if idx + k < max_token_len:
            new_pos[idx + k] = old_pos[-1] + k * delta / 4

    clip_model.context_length = max_token_len
    clip_model.positional_embedding = nn.Parameter(new_pos)
    attn_mask = clip_model.build_attention_mask()
    for block in clip_model.transformer.resblocks:
        if hasattr(block, "attn_mask"):
            block.attn_mask = attn_mask

# --- Pooling ---
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bsz = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), seq_lens]

# --- DDP gather for CLIP contrastive ---
def batch_align(fabric, x: torch.Tensor, grads: bool = True) -> torch.Tensor:
    """
    将各进程的特征做 all_gather 并展平为 (global_batch, ...).
    - fabric 为 None 或无 all_gather：直接返回 x。
    - grads=False 时：不做 gather（验证阶段通常无需全局对比），直接返回 x。
    - 兼容 Fabric 版本差异：优先尝试 sync_grads=True，失败则降级。
    """
    if fabric is None or not hasattr(fabric, "all_gather"):
        return x
    if not grads:
        return x
    try:
        y = fabric.all_gather(x, sync_grads=True)
    except TypeError:
        # 老版本没有 sync_grads 参数
        y = fabric.all_gather(x)
    # y 形状通常是 [world_size, batch, ...]
    try:
        y = y.contiguous()
        if y.dim() >= 3:
            y = y.view(y.shape[0] * y.shape[1], *y.shape[2:])
    except Exception:
        pass
    return y

# --- Loss ---
_cls_criterion = torch.nn.CrossEntropyLoss()

def clip_loss(sim: torch.Tensor) -> torch.Tensor:
    """
    对称对比损失：对 sim 和 sim.T 分别做 CE，再取均值。
    sim: [N, N] = (logit_scale * img @ txt^T)
    """
    gt = torch.arange(sim.shape[0], device=sim.device)
    loss_i2t = _cls_criterion(sim, gt)
    loss_t2i = _cls_criterion(sim.t(), gt)
    return (loss_i2t + loss_t2i) / 2.0

# --- Utils ---
def print_trainable_parameters(fabric, model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, p in model.named_parameters():
        n = p.numel()
        all_param += n
        if p.requires_grad:
            trainable_params += n
    msg = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / max(1, all_param):.2f}"
    if hasattr(fabric, "print"):
        fabric.print(msg)
        try:
            dev = fabric.device if hasattr(fabric, "device") else (next(model.parameters()).device)
            fabric.print(f'Current CUDA memory allocated: {torch.cuda.memory_allocated(device=dev)} bytes')
        except Exception:
            pass
    else:
        print(msg)
        try:
            dev = next(model.parameters()).device
            print(f'Current CUDA memory allocated: {torch.cuda.memory_allocated(device=dev)} bytes')
        except Exception:
            pass
