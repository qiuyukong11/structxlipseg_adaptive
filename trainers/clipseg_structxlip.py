import re

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import PVL_Adapter
from .scale_block import ScaleBlock
from utils.func import extend_clip_text_context

from clip import clip


COLOR_WORDS = [
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "orange", "purple", "pink", "brown", "beige", "cyan",
    "magenta", "turquoise", "teal", "maroon", "navy", "violet", "indigo", "gold", "silver", "ivory", "cream", "olive", "tan",
    "peach", "mint", "burgundy", "crimson", "scarlet", "lavender", "lilac", "azure", "aqua", "aquamarine", "navy blue",
    "sky blue", "baby blue", "light blue", "dark blue", "light green", "dark green", "forest green", "lime green",
    "light red", "dark red", "rose red", "wine red", "light pink", "hot pink", "dark gray", "light gray", "dark grey", "light grey",
]
MATERIAL_WORDS = [
    "cotton", "wool", "silk", "linen", "denim", "leather", "suede", "velvet", "satin", "chiffon",
    "polyester", "nylon", "spandex", "acrylic", "rayon", "cashmere", "fleece", "corduroy", "lace", "mesh",
    "canvas", "tweed", "felt", "rubber", "plastic", "metal", "steel", "iron", "aluminum", "bronze", "brass",
    "ceramic", "glass", "wood", "bamboo", "stone", "marble", "granite", "concrete", "clay", "paper",
    "fur", "shearling", "down", "feather", "denier", "foam",
]
TEXTURE_WORDS = [
    "smooth", "rough", "soft", "hard", "glossy", "matte", "shiny", "dull", "coarse", "fine",
    "grainy", "fuzzy", "fluffy", "silky", "velvety", "wrinkled", "crumpled", "woven", "knit",
    "striped", "plaid", "checkered", "polka dot", "dotted", "spotted", "paisley", "floral",
    "camouflage", "camo", "animal print", "zebra print", "leopard print", "snake print",
    "herringbone", "chevron", "geometric", "abstract", "tie-dye", "ombre", "gradient", "marbled",
    "transparent", "translucent", "opaque", "frosted", "sheer", "mesh", "netted",
]

_MATERIAL_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, MATERIAL_WORDS))), flags=re.IGNORECASE)
_TEXTURE_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, TEXTURE_WORDS))), flags=re.IGNORECASE)
_COLOR_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, COLOR_WORDS))), flags=re.IGNORECASE)


def remove_visual_words(text: str, remove_colors=False, remove_materials=False, remove_textures=False) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    original_text = text
    if remove_colors:
        text = _COLOR_RE.sub("", text)
    if remove_materials:
        text = _MATERIAL_RE.sub("", text)
    if remove_textures:
        text = _TEXTURE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"(,\s*)+,", ",", text)
    words = text.split()
    meaningful_words = [w for w in words if w.lower() not in ["and", "or", "a", "an", "the", ",", "."]]
    if len(meaningful_words) < 2 or not text.strip():
        return original_text
    return text


def clip_loss(sim):
    gt = torch.arange(sim.shape[0], dtype=torch.long, device=sim.device)
    return (F.cross_entropy(sim, gt) + F.cross_entropy(sim.t(), gt)) / 2.0

# ========= 自适应滑窗（全局 token 空间，避免错位） =========
def adaptive_sliding_token_spans_global(caption: str, base_window: int = 3, stride: int = 1, max_spans: int = 64, max_tokens: int = 77):
    """
    在“整句 token 空间”里返回可用的 (start_token, end_token) 跨度。
    关键：只 tokenizer(caption) 一次，然后用全局的 char_to_token 做映射，避免子句 tokenizer 造成的索引错位。
    """
    if not isinstance(caption, str) or not caption.strip():
        return []

    word_matches = list(re.finditer(r"\b\w+\b", caption))
    if len(word_matches) == 0:
        return []

    # 找句子边界（基于标点）
    sentence_bounds = []
    last = 0
    for m in re.finditer(r"[.,;!?]+", caption):
        sentence_bounds.append((last, m.end()))
        last = m.end()
    if last < len(caption):
        sentence_bounds.append((last, len(caption)))

    spans = []
    wi = 0
    for _, e_char in sentence_bounds:
        start_wi = wi
        while wi < len(word_matches) and word_matches[wi].start() < e_char:
            wi += 1
        end_wi = wi - 1
        seg_len = end_wi - start_wi + 1
        if seg_len <= 0:
            continue

        if seg_len <= 4:
            w = 2
        elif seg_len <= 10:
            w = base_window
        elif seg_len <= 20:
            w = base_window + 1
        elif seg_len <= 40:
            w = base_window + 2
        else:
            w = base_window + 3

        for i in range(start_wi, end_wi - w + 2, stride):
            start_char = word_matches[i].start()
            end_char = word_matches[i + w - 1].end()
            start_token = 1 + len(clip._tokenizer.encode(caption[:start_char]))
            end_token = start_token + len(clip._tokenizer.encode(caption[start_char:end_char]))
            start_token = max(1, min(start_token, max_tokens - 2))
            end_token = max(start_token + 1, min(end_token, max_tokens - 1))
            spans.append((start_token, end_token))
            if len(spans) >= max_spans:
                break
        if len(spans) >= max_spans:
            break

    return spans

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'StructXLIPSeg',
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    
    max_token_len = int(getattr(cfg.MODEL, "MAX_TOKEN_LEN", 77))
    if max_token_len > model.context_length:
        extend_clip_text_context(model, max_token_len)
        print(f"[CLIP] extended frozen text context to max_token_len={max_token_len}")
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, layer in enumerate(self.transformer.resblocks):
            x = self.transformer([x])[0]
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super(CustomCLIP, self).__init__()

        # === Core CLIP Components ===
        self.vision_model = clip_model.visual
        self.text_model = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.clip_model = clip_model

        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.text_emb_dim = 512
            self.patch_size = 16
            self.text_proj_dim = 512

        else:
            raise NotImplementedError("Other backbones not implemented yet.")

        self.temperature = cfg.MODEL.TEMPERATURE
        self.context_length = int(getattr(cfg.MODEL, "MAX_TOKEN_LEN", 77))

        self.dtype = clip_model.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.beta = cfg.MODEL.BETA
        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE

        self.gate_init = cfg.MODEL.GATE_INIT
        self.cfg = cfg

        # ==== StructXLIP-specific Configurations ===
        struct_cfg = getattr(cfg, "STRUCTXLIP", None)
        self.lambda_scribble_text = float(getattr(struct_cfg, "LAMBDA_STRUCTURE_TEXT", 0.0))
        self.lambda_rgb_scribble = float(getattr(struct_cfg, "LAMBDA_RGB_STRUCTURE_CONSISTENCY", 0.0))
        self.lambda_chunk = float(getattr(struct_cfg, "LAMBDA_CHUNK_ALIGN", 0.0))
        self.chunk_tau = float(getattr(struct_cfg, "CHUNK_TAU", 0.07))
        self.chunk_base_window = int(getattr(struct_cfg, "CHUNK_BASE_WINDOW", 3))
        self.chunk_stride = int(getattr(struct_cfg, "CHUNK_STRIDE", 1))
        self.chunk_max_spans = int(getattr(struct_cfg, "CHUNK_MAX_SPANS", 64))
        self.remove_colors = bool(getattr(struct_cfg, "REMOVE_COLORS", False))
        self.remove_materials = bool(getattr(struct_cfg, "REMOVE_MATERIALS", False))
        self.remove_textures = bool(getattr(struct_cfg, "REMOVE_TEXTURES", False))
        # ==== End of StructXLIP-specific Configurations ===
        
        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(in_channels_vis=self.embed_dim, in_channels_txt=self.text_emb_dim, adapter_channels=adapter_channels, 
                            beta=self.beta, gate_init=self.gate_init)
            for _ in range(len(self.fusion_stages))
        ])

    def encode_text_image(self, tokenized_prompts, prompts, image, return_image_embed=False, return_text_tokens=False):

        x_txt = prompts + self.text_model.positional_embedding.type(self.dtype)
        x_txt = x_txt.permute(1, 0, 2)  # NLD -> LND

        x_img = self.vision_model.conv1(image)  # shape = [*, width, grid, grid]
        B, C, H, W = x_img.shape
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_img = x_img.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_img = torch.cat(
            [self.vision_model.class_embedding.to(x_img.dtype) + torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
             x_img], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)

        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # NLD -> LND

        hidden_states = []

        for i, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks,self.text_model.transformer.resblocks)):

            if(i in self.fusion_stages):
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img.transpose(1,0), x_txt.transpose(1,0))
                x_img = x_img + vis_pvl.transpose(1,0)
                x_txt = x_txt + txt_pvl.transpose(1,0)

            x_img, hidden_states = block([x_img, hidden_states])
            x_txt = layer([x_txt])

            x_txt = x_txt[0]
        
        x_txt = x_txt.permute(1, 0, 2)  # LND -> NLD
        text_tokens = x_txt
        x_txt = self.text_model.ln_final(text_tokens).type(self.dtype)
        x_txt = x_txt[torch.arange(x_txt.shape[0], device=x_txt.device), tokenized_prompts.argmax(dim=-1)] @ self.text_model.text_projection

        hidden_states = torch.stack(hidden_states, dim=0) # (Num Layers, L, N, D)
        x_patch =  hidden_states[:, 1:hidden_states.shape[1], :, :] # Remove visual ctx and class token
        x_patch = x_patch.permute(0, 2, 1, 3)  # LND -> NLD

        x_patch = x_patch[-1]

        x_cls =  hidden_states[-1, 0, :, :] # class token
        
        x_patch = self.vision_model.ln_post(x_patch)
        x_cls = self.vision_model.ln_post(x_cls)
        x_patch = x_patch @ self.vision_model.proj
        
        # StructXLIP-specific
        # x_cls 是 ViT 最后层的 class token，可以理解成整张图的全局图像表示。
        x_cls = x_cls @ self.vision_model.proj

        outputs = [x_patch, x_txt]
        if return_text_tokens:
            #  每个文本 token 的特征，[B, token_len, 512]
            outputs.append(text_tokens)
        if return_image_embed:
            # 整张图的全局 image embedding，[B, 512]
            outputs.append(x_cls)
        return tuple(outputs)

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = F.normalize(text_features, dim=-1, eps=1e-6)
        seg_feats = F.normalize(image_features, dim=-1, eps=1e-6)

        h_patch = H // self.patch_size
        w_patch = W // self.patch_size
        seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2)

        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(text_features).unsqueeze(1), self.upscale(seg_feats)
        )
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear", align_corners=False).squeeze(1)
        return seg_logits

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _tokenize_texts(self, texts, device):
        return torch.cat([
            clip.tokenize(t, context_length=self.context_length, truncate=True) for t in texts
        ]).to(device)

    def compute_structxlip_losses(
        self,
        image,
        text,
        tokenized_prompts,
        text_prompts,
        text_features,
        text_tokens,
        image_embeds,
        structure_image=None,
        edge_images=None,
        has_structure=None,
        edge_valid_mask=None,
    ):
        device = image.device
        bs = image.shape[0]
        eps = 1e-8
        zero = image.new_zeros(())
        loss_st = zero  # 全局边缘图像和全局过滤文本
        loss_rs = zero  # 全局图像和全局边缘图像的一致性
        loss_chunk_align = zero  # 局部边缘图像和对应文本块的一致性

        # ---- 文本预处理：过滤文本 ----
        filtered_texts = [
            remove_visual_words(
                cap,
                remove_colors=self.remove_colors,
                remove_materials=self.remove_materials,
                remove_textures=self.remove_textures,
            )
            for cap in text
        ]

        if self.remove_colors or self.remove_materials or self.remove_textures:
            filtered_tokenized = self._tokenize_texts(filtered_texts, device)
            with torch.no_grad():
                filtered_prompts = self.clip_model.token_embedding(filtered_tokenized).type(self.dtype)
            _, filtered_text_features = self.encode_text_image(filtered_tokenized, filtered_prompts, image)
        else:
            filtered_text_features = text_features

        if structure_image is not None and (self.lambda_scribble_text != 0 or self.lambda_rgb_scribble != 0):
            if has_structure is None:
                has_org_s = torch.ones(bs, dtype=torch.bool, device=device)
            else:
                has_org_s = has_structure.to(device=device).view(-1).bool()

            if has_org_s.any():
                _, _, structure_embeds = self.encode_text_image(
                    tokenized_prompts, text_prompts, structure_image, return_image_embed=True
                )
                if self.lambda_scribble_text != 0:
                    img = F.normalize(structure_embeds[has_org_s] + eps, dim=-1)
                    txt = F.normalize(filtered_text_features[has_org_s] + eps, dim=-1)
                    loss_st = clip_loss(self.logit_scale.exp().detach() * (img @ txt.t()))
                if self.lambda_rgb_scribble != 0:
                    loss_rs = F.cosine_embedding_loss(
                        F.normalize(image_embeds[has_org_s] + eps, dim=-1),
                        F.normalize(structure_embeds[has_org_s] + eps, dim=-1),
                        torch.ones(has_org_s.sum(), device=device),
                    )

        if self.lambda_chunk != 0 and edge_images is not None and edge_valid_mask is not None:
            if edge_images.dim() == 5:
                k = edge_images.shape[1]
                edge_flat = edge_images.reshape(bs * k, *edge_images.shape[2:])
                edge_tokenized = tokenized_prompts[:, None, :].expand(bs, k, -1).reshape(bs * k, -1)
                edge_prompts = text_prompts[:, None, :, :].expand(bs, k, -1, -1).reshape(bs * k, text_prompts.shape[1], text_prompts.shape[2])
                _, _, edge_seg_embeds_flat = self.encode_text_image(
                    edge_tokenized, edge_prompts, edge_flat, return_image_embed=True
                )
                edge_all = F.normalize(edge_seg_embeds_flat + eps, dim=-1)
                valid_all_mask = edge_valid_mask.reshape(-1).to(device=device).bool()

                if valid_all_mask.any():
                    neg_inf = torch.finfo(edge_all.dtype).min
                    tau = float(max(1e-6, self.chunk_tau))
                    items_with_chunks = 0

                    for b in range(bs):
                        # 1) 全局对齐滑窗
                        chunk_token_spans = adaptive_sliding_token_spans_global(
                            filtered_texts[b],
                            base_window=self.chunk_base_window,
                            stride=self.chunk_stride,
                            max_spans=self.chunk_max_spans,
                            max_tokens=self.context_length,
                        )
                        if not chunk_token_spans:
                            continue

                        text_tokens_b = text_tokens[b]
                        chunk_embeddings = []
                        for s, e in chunk_token_spans:
                            if 0 <= s < e <= text_tokens_b.shape[0]:
                                # 用范数做 soft 权重（比简单 mean 更稳）
                                with torch.no_grad():
                                    w = F.softmax(text_tokens_b[s:e].norm(dim=-1), dim=0) # [L]
                                chunk_embeddings.append((text_tokens_b[s:e] * w[:, None]).sum(dim=0))
                        if not chunk_embeddings:
                            continue

                        chunk_embeds = torch.stack(chunk_embeddings, dim=0)  # [Nc, H]
                        chunk_embeds = self.text_model.ln_final(chunk_embeds).type(self.dtype)
                        chunk_embeds = chunk_embeds @ self.text_model.text_projection  # [Nc, D]
                        chunk_embeds_n = F.normalize(chunk_embeds, dim=-1)  # [Nc, D]

                        # 2) 相似度（限制有效温度，防止过尖）
                        scale_eff = (self.logit_scale.exp() / tau).clamp(max=100.0)
                        sim_all = scale_eff * (chunk_embeds_n @ edge_all.t())  # [Nc, B*K]
                        
                        # 3) 分母（所有有效 edge）
                        sim_den = sim_all.masked_fill(~valid_all_mask.unsqueeze(0), neg_inf)
                        log_den = torch.logsumexp(sim_den, dim=1)  # [Nc]

                        # 4) 正样本（本样本的 Top-K 有效 edge）
                        pos_mask_row = torch.zeros(bs * k, dtype=torch.bool, device=device)
                        kb = edge_valid_mask[b].to(device=device).bool()  # [K]
                        if kb.any():
                            idx_start = b * k
                            pos_idx = torch.arange(idx_start, idx_start + k, device=device)[kb]
                            pos_mask_row[pos_idx] = True
                        else:
                            continue

                        sim_pos = sim_all.masked_fill(~pos_mask_row.unsqueeze(0), neg_inf)
                        log_pos = torch.logsumexp(sim_pos, dim=1)  # [Nc]
                        
                        # 5) Multi-Positive InfoNCE
                        loss_chunk_align = loss_chunk_align - (log_pos - log_den).mean()
                        items_with_chunks += 1

                    if items_with_chunks > 0:
                        loss_chunk_align = loss_chunk_align / items_with_chunks

        self.last_structxlip_losses = {
            "loss_st": loss_st.detach(),
            "loss_rs": loss_rs.detach(),
            "loss_chunk_align": loss_chunk_align.detach(),
        }
        return (
            self.lambda_scribble_text * loss_st
            + self.lambda_rgb_scribble * loss_rs
            + self.lambda_chunk * loss_chunk_align
        )

    def forward(self, image, text, num_samples=30, return_clip_loss=True,
        structure_image=None,
        edge_images=None,
        has_structure=None,
        edge_valid_mask=None,
        original_text=None,
    ):
        B, C, H, W = image.shape
        tokenized_prompts = self._tokenize_texts(text, image.device)

        with torch.no_grad():
            text_prompts = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        need_aux_features = self.training and return_clip_loss
        if need_aux_features:
            image_features, text_features, text_tokens, image_embeds = self.encode_text_image(
                tokenized_prompts, text_prompts, image, return_text_tokens=True, return_image_embed=True
            )
        else:
            image_features, text_features = self.encode_text_image(
                tokenized_prompts, text_prompts, image
            )

        seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)

        if self.training:
            if not return_clip_loss:
                return seg_logits, seg_logits.new_zeros(())

            patch_logits = F.normalize(image_features, dim=-1, eps=1e-6)
            patch_mean = patch_logits.mean(dim=1)  # shape: (B, D)

            # Compute logits
            logits_per_image = (patch_mean @ text_features.T) / self.temperature   # (B, B)
            logits_per_text = (text_features @ patch_mean.T) / self.temperature  # (B, B)

            # --- Soft targets based on text similarity ---
            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature # (B, B)
                text_sim = F.normalize(text_sim, dim=-1, eps=1e-6)
                soft_targets = F.softmax(text_sim, dim=-1)  # temperature-controlled soft labels

            loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
            loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)

            clip_aux_loss = (loss_i2t + loss_t2i) / 2

            # === StructXLIP auxiliary losses ===
            struct_text = original_text if original_text is not None else text
            if struct_text == text:
                struct_tokenized_prompts = tokenized_prompts
                struct_text_prompts = text_prompts
                struct_text_features = text_features
                struct_text_tokens = text_tokens
                struct_image_embeds = image_embeds
            else:
                # struct_text 全局文本描述, image 是全局图像输入
                struct_tokenized_prompts = self._tokenize_texts(struct_text, image.device)
                with torch.no_grad():
                    struct_text_prompts = self.clip_model.token_embedding(struct_tokenized_prompts).type(self.dtype)
                _, struct_text_features, struct_text_tokens, struct_image_embeds = self.encode_text_image(
                    struct_tokenized_prompts,
                    struct_text_prompts,
                    image,
                    return_text_tokens=True,
                    return_image_embed=True,
                )

            # 输入：全局图像，全局文本，全局边缘图，局部边缘图，局部文本
            structxlip_loss = self.compute_structxlip_losses(
                image=image,
                text=struct_text,
                tokenized_prompts=struct_tokenized_prompts,
                text_prompts=struct_text_prompts,
                text_features=struct_text_features,
                text_tokens=struct_text_tokens,
                image_embeds=struct_image_embeds,
                structure_image=structure_image,
                edge_images=edge_images,
                has_structure=has_structure,
                edge_valid_mask=edge_valid_mask,
            )

            return seg_logits, clip_aux_loss, structxlip_loss
            # === End of training-time auxiliary loss calculation ===
        
        else:

            seg_samples = []
            for _ in range(num_samples):

                image_features, text_features = self.encode_text_image(
                    tokenized_prompts, text_prompts, image
                )

                seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)
        
                seg_samples.append(seg_logits)

            seg_samples = torch.stack(seg_samples, dim=0)  # [N, B, C, H, W]

            return seg_samples 

def build_clipseg_structxlip(cfg):

    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)

    return model

# Backward-compatible alias for older scripts that used the clip builder name.
build_clipseg_clip = build_clipseg_structxlip
