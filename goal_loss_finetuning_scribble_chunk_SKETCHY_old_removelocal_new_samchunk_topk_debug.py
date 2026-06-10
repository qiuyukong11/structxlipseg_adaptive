import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math, json, argparse, random, re
from pathlib import Path
from PIL import Image
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import transformers
from torch.utils.data import Dataset
import wandb
# import spacy  # 可选，不强制使用

from utils.func import *  # longclip_pos_embeddings, batch_align, print_trainable_parameters 等
# from utils.transforms import *  # 未使用
import numpy as np
from adaptive_loss_weights import AdaptiveWeightState, get_loss_weights

# =========（保留）辅助：若后续需要 patch-level，可继续使用 =========
def get_positive_patch_indices_from_scribble(scribble_image: torch.Tensor, patch_size=16):
    scribble_np = scribble_image.permute(1, 2, 0).cpu().numpy()
    scribble_np_norm = (scribble_np - scribble_np.min()) / (scribble_np.max() - scribble_np.min() + 1e-8)
    non_white_mask = (scribble_np_norm.mean(axis=-1) < 0.98)
    if not non_white_mask.any():
        return []
    h, w = non_white_mask.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    positive_indices = set()
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            if non_white_mask[y0:y1, x0:x1].any():
                positive_indices.add(i * num_patches_w + j)
    return sorted(list(positive_indices))

def generate_sliding_window_token_spans(caption: str, tokenizer, window_size: int, stride: int = 1):
    """
    旧版固定窗口函数（保留以备需要）；当前训练改用自适应窗口。
    """
    encoding = tokenizer(caption, return_offsets_mapping=True, add_special_tokens=True)
    words_with_indices = []
    for match in re.finditer(r'\b\w+\b', caption.lower()):
        words_with_indices.append((match.group(0), match.start()))
    if len(words_with_indices) < window_size:
        return []
    token_spans = []
    for i in range(0, len(words_with_indices) - window_size + 1, stride):
        start_char = words_with_indices[i][1]
        end_word, end_char_start = words_with_indices[i + window_size - 1]
        end_char = end_char_start + len(end_word)
        start_token = encoding.char_to_token(start_char)
        end_token = encoding.char_to_token(end_char - 1)
        if start_token is not None and end_token is not None and end_token >= start_token:
            token_spans.append((start_token, end_token + 1))
    return token_spans

# ========= 环境 & 精度 =========
torch.autograd.set_detect_anomaly(True)
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# ========= 文本指针工具 =========
def _build_char2tok(text, tokenizer, max_len_wo_special):
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                        truncation=True, max_length=max_len_wo_special, return_tensors=None)
    except Exception:
        return {}, []
    offsets = enc.get("offset_mapping", [])
    char2tok = {}
    for tid, offset in enumerate(offsets):
        if offset is not None and len(offset) == 2:
            s, e = offset
            for ch in range(s, e):
                char2tok[ch] = tid
    return char2tok

def _char_span_to_hidden_span(start_char, end_char, char2tok, max_len_hidden, bos_shift=1):
    if end_char <= start_char:
        return None
    t_start = char2tok.get(start_char)
    t_end_inclusive = char2tok.get(end_char - 1)
    if t_start is None or t_end_inclusive is None:
        return None
    start = max(1, min(t_start + bos_shift, max_len_hidden - 2))
    end = max(start + 1, min(t_end_inclusive + 1 + bos_shift, max_len_hidden - 1))
    return start, end

# ========= 去色 & 过滤 =========
COLOR_WORDS = [
    "red","blue","green","yellow","black","white","gray","grey","orange","purple","pink","brown","beige","cyan",
    "magenta","turquoise","teal","maroon","navy","violet","indigo","gold","silver","ivory","cream","olive","tan",
    "peach","mint","burgundy","crimson","scarlet","lavender","lilac","azure","teal","aqua","aquamarine","navy blue",
    "sky blue","baby blue","light blue","dark blue","light green","dark green","forest green","lime green",
    "light red","dark red","rose red","wine red","light pink","hot pink","dark gray","light gray","dark grey","light grey"
]
MATERIAL_WORDS = [
    "cotton", "wool", "silk", "linen", "denim", "leather", "suede", "velvet", "satin", "chiffon",
    "polyester", "nylon", "spandex", "acrylic", "rayon", "cashmere", "fleece", "corduroy", "lace", "mesh",
    "canvas", "tweed", "felt", "rubber", "plastic", "metal", "steel", "iron", "aluminum", "bronze", "brass",
    "ceramic", "glass", "wood", "bamboo", "stone", "marble", "granite", "concrete", "clay", "paper",
    "fur", "shearling", "down", "feather", "denier", "foam"
]
TEXTURE_WORDS = [
    "smooth", "rough", "soft", "hard", "glossy", "matte", "shiny", "dull", "coarse", "fine",
    "grainy", "fuzzy", "fluffy", "silky", "velvety", "wrinkled", "crumpled", "woven", "knit",
    "striped", "plaid", "checkered", "polka dot", "dotted", "spotted", "paisley", "floral",
    "camouflage", "camo", "animal print", "zebra print", "leopard print", "snake print",
    "herringbone", "chevron", "geometric", "abstract", "tie-dye", "ombre", "gradient", "marbled",
    "transparent", "translucent", "opaque", "frosted", "sheer", "mesh", "netted"
]
INSECT_WORDS = [
    # -----------------------------
    # BASIC COLOR TERMS
    # -----------------------------
    "black", "brown", "dark brown", "light brown", "tan", "beige", "cream",
    "white", "off-white", "gray", "grey", "charcoal", "slate", "ash",

    "red", "reddish", "orange", "yellow", "green", "blue", "purple",
    "pink", "magenta", "violet",

    # Common naturalistic insect colors
    "rust", "russet", "chestnut", "mahogany", "clay", "ochre", "umber",
    "sienna", "tawny", "fawn",

    "amber", "honey-colored", "golden", "bronze", "coppery",

    # Low-saturation color descriptors
    "pale", "dusky", "washed-out", "faded", "dim", "drab",

    # -----------------------------
    # SPECIALIZED ENTOMOLOGY COLORS
    # (frequently used in field guides)
    # -----------------------------
    "rufous", "testaceous", "fulvous", "ferruginous",
    "castaneous", "fuscous", "livid", "piceous",
    "violaceous", "cyaneous", "glaucous",

    # -----------------------------
    # NON-GEOMETRIC PIGMENTATION QUALITIES
    # -----------------------------
    "mottled",
    "blotchy",
    "flecked",
    "freckled",
    "stained",
    "tinged",
    "tinted",
    "smudged",
    "clouded",
    "diffuse",
    "suffused",
    "irregularly pigmented",
    "unevenly pigmented",
    "faintly pigmented",
    "deeply pigmented",
    "melanized",
    "depigmented",
    "discolored",
    "frosted",
    "pruinose",   # waxy bloom, common in insects
    "powdery",
    "mealy",
    "chalky",
    "dusty",
    "granular",
    "velvety",

    # Gradients & tonal transitions (not geometric patterns)
    "shaded",
    "darkened",
    "lightened",
    "somber",
    "sooty",
    "smoky",
    "smeared",

    # -----------------------------
    # IRIDESCENCE & STRUCTURAL COLORS
    # -----------------------------
    "iridescent",
    "metallic",
    "submetallic",
    "opalescent",
    "pearlescent",
    "rainbowlike",
    "prismatic",
    "lustrous",
    "holographic",   # used in beetle elytra descriptions

    # -----------------------------
    # SHEEN / SURFACE REFLECTANCE
    # -----------------------------
    "shiny",
    "glossy",
    "subglossy",
    "dull",
    "matte",
    "satiny",
    "silky",
    "polished",
    "reflective",
    "non-reflective",
    "lustrous",
    "sheeny",
    "mirrorlike",

    # -----------------------------
    # TRANSPARENCY & LIGHT TRANSMISSION
    # -----------------------------
    "transparent",
    "translucent",
    "semi-translucent",
    "opaque",
    "hyaline",     # glasslike, common for insect wings
    "subhyaline",
    "smoky-hyaline",

    # -----------------------------
    # COLOR TEMPERATURE & QUALITIES
    # -----------------------------
    "warm-toned",
    "cool-toned",
    "earthy",
    "vivid",
    "dull-colored",
    "bright",
    "pale-colored",
    "dark-colored",

    # -----------------------------
    # SURFACE TEXTURE (optical, not geometric)
    # -----------------------------
    "satiny",
    "silken",
    "glassy",
    "resinous",
    "lacquered",
    "oily",
    "greasy",
    "waxy",
    "glistening",
    "gleaming"
]

_MATERIAL_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, MATERIAL_WORDS))), flags=re.IGNORECASE)
_TEXTURE_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, TEXTURE_WORDS))), flags=re.IGNORECASE)
_COLOR_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, COLOR_WORDS))), flags=re.IGNORECASE)
_INSECT_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, INSECT_WORDS))), flags=re.IGNORECASE)
def remove_visual_words(text: str, remove_insect=False,remove_colors=False, remove_materials=False, remove_textures=False) -> str:
    """根据开关移除颜色 / 材质 / 纹理词汇"""
    if not isinstance(text, str) or not text.strip():
        return text
    original_text = text
    if remove_insect:
        text = _INSECT_RE.sub("", text)
    if remove_colors:
        text = _COLOR_RE.sub("", text)
    if remove_materials:
        text = _MATERIAL_RE.sub("", text)
    if remove_textures:
        text = _TEXTURE_RE.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'(,\s*)+,', ',', text)
    words = text.split()
    meaningful_words = [w for w in words if w.lower() not in ["and","or","a","an","the",",","."]]
    if len(meaningful_words) < 2 or not text.strip():
        return original_text
    return text

def first_existing_path(item, *keys):
    for key in keys:
        path = item.get(key)
        if path and os.path.exists(path):
            return path
    return None

def cosine_anneal_warm_decay(base, epoch, *, warm=1, decay_start=3, decay_end=8, floor=0.2):
    if epoch < warm:
        w = (epoch + 1) / max(1, warm)
    elif epoch < decay_start:
        w = 1.0
    else:
        T = max(1e-6, decay_end - decay_start)
        t = min(max(epoch - decay_start, 0.0), T)
        w = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t / T))
    return base * w

def clip_loss(sim):
    gt = torch.arange(len(sim), dtype=torch.long, device=sim.device)
    return (F.cross_entropy(sim, gt) + F.cross_entropy(sim.t(), gt)) / 2.0

def get_patch_tokens_from_bbox(patch_tokens, bbox, b, org_size, image_size=224, patch_size=16):
    org_width, org_height = org_size
    if org_width == 0 or org_height == 0: return patch_tokens.mean(dim=1)
    x1 = int(round(bbox['x1'][b].item() * image_size / org_width))
    y1 = int(round(bbox['y1'][b].item() * image_size / org_height))
    x2 = int(round(bbox['x2'][b].item() * image_size / org_width))
    y2 = int(round(bbox['y2'][b].item() * image_size / org_height))
    x1=max(0,min(x1,image_size-1)); y1=max(0,min(y1,image_size-1)); x2=max(0,min(x2,image_size)); y2=max(0,min(y2,image_size))
    patch_x1=x1//patch_size; patch_y1=y1//patch_size; patch_x2=(x2+patch_size-1)//patch_size; patch_y2=(y2+patch_size-1)//patch_size
    num_patches_w = image_size // patch_size
    indices=[]
    for i in range(patch_y1, patch_y2):
        for j in range(patch_x1, patch_x2):
            indices.append(i * num_patches_w + j + 1)  # +1 for CLS
    if not indices:
        return patch_tokens.mean(dim=1)
    relevant_tokens=patch_tokens[:, indices, :]
    return torch.mean(relevant_tokens, dim=1)

def get_text_tokens_from_segment(text_tokens, org_text, seg_text, processor):
    max_len_hidden = text_tokens.shape[1]
    org_text_clean = ' '.join(org_text.split()).strip()
    seg_text_clean = ' '.join(seg_text.split()).strip()
    seg_pos = org_text_clean.find(seg_text_clean)
    if seg_pos == -1:
        import difflib
        sentences = [s.strip() for s in re.split(r'[;,.]', org_text_clean) if s.strip()]
        best_match, best_ratio = None, 0.0
        for snt in sentences:
            ratio = difflib.SequenceMatcher(None, seg_text_clean, snt).ratio()
            if ratio > best_ratio:
                best_ratio, best_match = ratio, snt
        if best_match:
            start_char = org_text_clean.find(best_match)
            end_char = start_char + len(best_match)
        else:
            return text_tokens[:, 1:-1, :].mean(dim=1)
    else:
        start_char = seg_pos
        end_char = seg_pos + len(seg_text_clean)
    try:
        char2tok = _build_char2tok(org_text_clean, processor.tokenizer, max_len_wo_special=max_len_hidden-2)
        span = _char_span_to_hidden_span(start_char, end_char, char2tok, max_len_hidden, bos_shift=1)
        if span is None:
            return text_tokens[:, 1:-1, :].mean(dim=1)
        s, e = span
        return text_tokens[:, s:e, :].mean(dim=1)
    except Exception:
        return text_tokens[:, 1:-1, :].mean(dim=1)

# ========= 自适应滑窗（全局 token 空间，避免错位） =========
def adaptive_sliding_token_spans_global(caption: str, tokenizer, base_window: int = 3, stride: int = 1, max_spans: int = 64):
    """
    在“整句 token 空间”里返回可用的 (start_token, end_token) 跨度。
    关键：只 tokenizer(caption) 一次，然后用全局的 char_to_token 做映射，避免子句 tokenizer 造成的索引错位。
    """
    if not isinstance(caption, str) or not caption.strip():
        return []

    enc = tokenizer(caption, return_offsets_mapping=True, add_special_tokens=True)
    word_matches = list(re.finditer(r'\b\w+\b', caption))
    if len(word_matches) == 0:
        return []

    # 找句子边界（基于标点）
    sentence_bounds = []
    last = 0
    for m in re.finditer(r'[.,;!?]+', caption):
        sentence_bounds.append((last, m.end()))
        last = m.end()
    if last < len(caption):
        sentence_bounds.append((last, len(caption)))

    spans = []
    wi = 0
    for (s_char, e_char) in sentence_bounds:
        start_wi = wi
        while wi < len(word_matches) and word_matches[wi].start() < e_char:
            wi += 1
        end_wi = wi - 1
        seg_len = end_wi - start_wi + 1
        if seg_len <= 0:
            continue

        # 自适应窗口
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
            t0 = enc.char_to_token(start_char)
            t1 = enc.char_to_token(end_char - 1)
            if t0 is not None and t1 is not None and t1 >= t0:
                spans.append((t0, t1 + 1))
            if len(spans) >= max_spans:
                break
        if len(spans) >= max_spans:
            break

    return spans

# ========= 数据集 =========
class DLoader(Dataset):
    def __init__(self, data_list, processor, new_max_token, chunk_top_k: int):
        self.data_list = data_list
        self.processor = processor
        self.new_max_token = new_max_token
        self.chunk_top_k = chunk_top_k

    def __len__(self):
        return len(self.data_list)

    def _load_image(self, path, mode='RGB'):
        if not path or not os.path.exists(path): return None, (0, 0)
        try:
            img = Image.open(path).convert(mode)
            return img, img.size
        except Exception as e:
            print(f"Warning: Error loading image {path}: {e}")
            return None, (0, 0)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        org_photo, org_photo_size = self._load_image(item["original_filename"], mode='RGB')
        org_caption = item["original_caption"]
        # llm_caption = item["qwen3_caption"]
        llm_caption = item["original_caption"]
        # 一行版（任一存在就用）
        # org_sketch_path = item.get("original_filename_canny_sketch") or item.get("original_filename_canny")
        # org_sketch, _ = self._load_image(org_sketch_path, mode="L")

        # org_sketch, _ = self._load_image(item.get("original_filename_canny_sketch"), mode='L')
        org_sketch_path = first_existing_path(
            item,
            "original_filename_diffusion",
            "original_filename_diffusion_sketch",
            "original_filename_sketch",
            "original_filename_canny_sketch",
            "original_filename_canny",
        )
        org_sketch, _ = self._load_image(org_sketch_path, mode='L')
        # org_sketch, _ = self._load_image(item.get("original_filename_canny_sketch"), mode='L')
        if org_sketch is None:
            print(f"{item['original_filename']} org_sketchy is None")

        # 主 segment（保持原逻辑，便于其它损失）
        segments = item.get("segment", [])
        if not segments:
            segments = [{"similarity_score": 0.0, "filename": None, "caption": "",
                         "bbox_coordinates": {'x1':0,'y1':0,'x2':0,'y2':0,'width':0,'height':0}}]
        segment = max(segments, key=lambda x: x.get("similarity_score", 0.0))
        seg_photo, _ = self._load_image(segment.get("filename"), mode='RGB')
        seg_caption = segment.get("caption", "")
        # seg_sketch, _ = self._load_image(
        # segment.get("filename_canny_sketch") or segment.get("filename_sketch_canny"),
        # mode="L",
        # )

        # seg_sketch, _ = self._load_image(segment.get("filename_canny_sketch"), mode='L')
        # seg_sketch, _ = self._load_image(segment.get("filename_sketch_canny"), mode='L')
        seg_sketch_path = first_existing_path(
            segment,
            "filename_diffusion_cropped",
            "filename_diffusion_sketch",
            "filename_sketch",
            "filename_canny_sketch",
            "filename_sketch_canny",
        )
        seg_sketch, _ = self._load_image(seg_sketch_path, mode='L')
        bbox = segment.get("bbox_coordinates", {'x1':0,'y1':0,'x2':0,'y2':0,'width':0,'height':0})

        # 取 Top-K segments 的 canny 作为多正样本
        edge_segments = sorted(item.get("segment", []),
                               key=lambda x: x.get("similarity_score", float("-inf")),
                               reverse=True)
        
        top_segments = edge_segments[:self.chunk_top_k]

        dummy_rgb = Image.new('RGB',
                              (self.processor.image_processor.crop_size['height'],
                               self.processor.image_processor.crop_size['width']), (0, 0, 0))
        
        org_data = self.processor(images=org_photo if org_photo else dummy_rgb, text=org_caption,
                                  return_tensors="pt", truncation=True, padding="max_length", max_length=self.new_max_token)
        seg_data = self.processor(images=seg_photo if seg_photo else dummy_rgb, text=seg_caption,
                                  return_tensors="pt", truncation=True, padding="max_length", max_length=self.new_max_token)

        # 原图与 seg 的 scribble
        if org_sketch:
            org_scribble_pixels = self.processor(images=org_sketch.convert("RGB"), return_tensors="pt").pixel_values[0]
            has_org_scribble = torch.tensor(True)
        else:
            org_scribble_pixels = self.processor(images=dummy_rgb, return_tensors="pt").pixel_values[0]
            has_org_scribble = torch.tensor(False)

        if seg_sketch:
            seg_scribble_pixels = self.processor(images=seg_sketch.convert("RGB"), return_tensors="pt").pixel_values[0]
            has_seg_scribble = torch.tensor(True)
        else:
            seg_scribble_pixels = self.processor(images=dummy_rgb, return_tensors="pt").pixel_values[0]
            has_seg_scribble = torch.tensor(False)

        # Top-K edge canny：组装为 [K, 3, H, W]，并返回有效掩码 [K]
        edge_imgs, valid_flags = [], []
        for k in range(self.chunk_top_k):
            path = None
            if k < len(top_segments):
                # 一行版filename_sketch_canny
                # path = top_segments[k].get("filename_canny_sketch") or top_segments[k].get("filename_sketch_canny")

                # path = top_segments[k].get("filename_canny_sketch")
                # print('有没有路径',path)
                
                # if not os.path.exists(path) and path.lower().endswith(".png"):
                #     alt_path = path[:-4] + ".jpg"
                #     if os.path.exists(alt_path):
                #         path = alt_path
                        
                # path = top_segments[k].get("filename_canny_sketch")
                path = first_existing_path(
                    top_segments[k],
                    "filename_diffusion_cropped",
                    "filename_diffusion_sketch",
                    "filename_sketch",
                    "filename_canny_sketch",
                    "filename_sketch_canny",
                )
                
            if path and os.path.exists(path):
                im, _ = self._load_image(path, mode='L')
                edge_imgs.append(im.convert("RGB"))
                
                valid_flags.append(True)
            else:
                edge_imgs.append(dummy_rgb)
                valid_flags.append(False)

        edge_scribble_pixels = torch.stack([
            self.processor(images=img, return_tensors="pt").pixel_values[0] for img in edge_imgs
        ], dim=0)  # [K, 3, H, W]
        edge_valid_mask = torch.tensor(valid_flags, dtype=torch.bool)  # [K]

        return (org_data.pixel_values[0], org_data.input_ids[0],
                seg_data.pixel_values[0], seg_data.input_ids[0],
                bbox, org_caption, seg_caption,llm_caption,
                org_scribble_pixels, seg_scribble_pixels,
                has_org_scribble, has_seg_scribble, org_photo_size,
                edge_scribble_pixels, edge_valid_mask)

# ========= 主流程 =========
def main(args):
    wandb.init(
    project=args.wandb_project,
    name=getattr(args, "wandb_run_name", None),
    config=vars(args)
)
    fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto", precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(args.seed)

    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset) as f:
        train_list = json.load(f)

    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.CLIPModel.from_pretrained(args.model)
        longclip_pos_embeddings(model, args.new_max_token)
        print_trainable_parameters(fabric, model)
        
        # # ================== (新增) FLOPs 计算模块 ==================
        # if fabric.global_rank == 0:
        #     fabric.print("\n--- 正在计算模型 FLOPs ---")
        #     try:
        #         # 1. 准备 Vision Model 的 dummy input
        #         dummy_pixel_values = torch.randn(
        #             1, 3, args.image_size, args.image_size, 
        #             device=fabric.device
        #         )
        #         vision_flops = FlopCountAnalysis(model.vision_model, dummy_pixel_values)
                
        #         # 2. 准备 Text Model 的 dummy input
        #         dummy_input_ids = torch.randint(
        #             0, model.config.text_config.vocab_size, 
        #             (1, args.new_max_token), 
        #             device=fabric.device
        #         )
        #         text_flops = FlopCountAnalysis(model.text_model, dummy_input_ids)
                
        #         total_flops_val = vision_flops.total() + text_flops.total()
                
        #         # 3. 打印详细结果
        #         fabric.print("--- Vision Model FLOPs (详细) ---")
        #         fabric.print(flop_count_table(vision_flops))
        #         fabric.print("--- Text Model FLOPs (详细) ---")
        #         fabric.print(flop_count_table(text_flops))
                
        #         # 4. 打印总结
        #         gflops_vision = vision_flops.total() / 1e9
        #         gflops_text = text_flops.total() / 1e9
        #         gflops_total = total_flops_val / 1e9
                
        #         fabric.print(f"\n--- 总结 (GFLOPs, Batch Size=1) ---")
        #         fabric.print(f"Vision Model: {gflops_vision:.2f} GFLOPs")
        #         fabric.print(f"Text Model:   {gflops_text:.2f} GFLOPs")
        #         fabric.print(f"Total (V+T):  {gflops_total:.2f} GFLOPs")
        #         fabric.print("------------------------------------")

        #         # 5. (可选) 记录到 wandb
        #         if wandb.run:
        #             wandb.config.update({
        #                 "gflops_vision": gflops_vision,
        #                 "gflops_text": gflops_text,
        #                 "gflops_total": gflops_total
        #             })

        #     except ImportError:
        #         fabric.print("计算 FLOPs 失败: 未找到 'fvcore' 库。")
        #         fabric.print("请运行: pip install fvcore")
        #     except Exception as e:
        #         fabric.print(f"计算 FLOPs 时出错: {e}")
        # # ================== (新增) 计算结束 ==================

    dataset_train = DLoader(train_list, processor, args.new_max_token, chunk_top_k=args.chunk_top_k)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True, shuffle=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    
    train(fabric, model, optimizer, train_loader, processor, args)

# ========= 训练 =========
def train(fabric: L.Fabric, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader, processor, args) -> None:
    it = 0
    total_it = len(train_loader) * args.epochs
    mse_loss = torch.nn.MSELoss()
    adaptive_state = AdaptiveWeightState(beta=args.adaptive_ema_beta)
    base_loss_weights = {
        "scribble_text": args.lambda_scribble_text,
        "rgb_scribble": args.lambda_rgb_scribble_consistency,
        "chunk": args.lambda_chunk,
    }
    if fabric.global_rank == 0:
        fabric.print(f"Adaptive weight strategy: {args.adaptive_weight_strategy}")

    for epoch in range(args.epochs):
        for i, samples in enumerate(train_loader):
            lr = (args.init_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * it / total_it)) + args.min_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            (org_image, org_text, seg_image, seg_text,
             bbox, org_caption, seg_caption,llm_caption,
             org_scribble_pixels, seg_scribble_pixels,
             has_org_scribble, has_seg_scribble,
             org_image_size,
             edge_scribble_pixels, edge_valid_mask) = samples   # edge_scribble_pixels: [B, K, 3, H, W]; mask: [B, K]

            bs = org_image.shape[0]
            K = edge_scribble_pixels.shape[1]
            eps = 1e-8

            # ---- 文本预处理 ----
            filtered_texts_org = [
                remove_visual_words(cap,
                    remove_colors=args.remove_colors,
                    remove_materials=args.remove_materials,
                    remove_textures=args.remove_textures
                )
                for cap in llm_caption
            ]
            filtered_texts_seg = [
                remove_visual_words(cap,
                    remove_colors=args.remove_colors,
                    remove_materials=args.remove_materials,
                    remove_textures=args.remove_textures
                )
                for cap in llm_caption
            ]
            enc_org_filtered = processor(text=filtered_texts_org, return_tensors="pt", padding="max_length", truncation=True, max_length=args.new_max_token).input_ids.to(fabric.device)
            enc_seg_filtered = processor(text=filtered_texts_seg, return_tensors="pt", padding="max_length", truncation=True, max_length=args.new_max_token).input_ids.to(fabric.device)
            
            # ---- 前向：图像与文本拼接 ----
            # 图像顺序：org | seg | org_scribble | seg_scribble | edge_topK(flatten)
            edge_flat = edge_scribble_pixels.reshape(bs * K, *edge_scribble_pixels.shape[2:])  # [B*K, 3, H, W]
            images_cat = torch.cat((org_image, seg_image, org_scribble_pixels, seg_scribble_pixels, edge_flat), dim=0)
            texts_cat = torch.cat((org_text, seg_text, enc_org_filtered, enc_seg_filtered), dim=0)
            
            outputs = model(pixel_values=images_cat, input_ids=texts_cat, output_hidden_states=True, return_dict=True)
            vision_outputs, text_outputs = outputs.vision_model_output, outputs.text_model_output
            image_embeds_all, text_embeds_all = outputs.image_embeds, outputs.text_embeds
            
            # 按数量切分图像嵌入
            p0, p1, p2, p3 = 0, bs, 2*bs, 3*bs
            p4, p5 = 4*bs, 4*bs + bs*K
            org_image_embeds      = image_embeds_all[p0:p1]
            seg_image_embeds      = image_embeds_all[p1:p2]
            org_scribble_embeds   = image_embeds_all[p2:p3]
            seg_scribble_embeds   = image_embeds_all[p3:p4]
            edge_seg_embeds_flat  = image_embeds_all[p4:p5]                                # [B*K, D]
            edge_seg_embeds       = edge_seg_embeds_flat.view(bs, K, -1)                   # [B, K, D]

            # 文本嵌入切分
            org_text_embeds, seg_text_embeds, filtered_org_text_embeds, filtered_seg_text_embeds = torch.chunk(text_embeds_all, 4)

            # token 级特征（用于其它损失）
            # all_patch_tokens = vision_outputs.hidden_states[-1]
            all_patch_tokens = vision_outputs.last_hidden_state

            org_patch_tokens = all_patch_tokens[:bs]
            # all_patch_tokens = text_outputs.last_hidden_state
            # org_text_tokens = text_outputs.hidden_states[-1][:bs]
            org_text_tokens = text_outputs.last_hidden_state[:bs]
            logit_scale_main = model.logit_scale.exp()
            logit_scale_aux = logit_scale_main.detach()

            # ---- 基础 CLIP 损失 ----
            x_i_org_seg = batch_align(fabric, F.normalize(torch.cat((org_image_embeds, seg_image_embeds), dim=0) + eps, dim=-1))
            x_t_org_seg = batch_align(fabric, F.normalize(torch.cat((org_text_embeds,  seg_text_embeds),  dim=0) + eps, dim=-1))
            x_i_org, x_i_seg = x_i_org_seg.chunk(2)
            x_t_org, x_t_seg = x_t_org_seg.chunk(2)
            sim_org = logit_scale_main * x_i_org @ x_t_org.t()
            loss_org = clip_loss(sim_org)
            sim_seg = logit_scale_main * x_i_seg @ x_t_seg.t()
            loss_seg = clip_loss(sim_seg)

            # ---- Patch-BBox 对齐（保持）----
            mse_loss = torch.nn.MSELoss()
            patch_pooled = torch.cat([
                get_patch_tokens_from_bbox(
                    org_patch_tokens[b:b+1], bbox, b,
                    (org_image_size[0][b].item(), org_image_size[1][b].item()),
                    image_size=args.image_size, patch_size=16
                ) for b in range(bs)
            ], dim=0)
            patch_pooled = model.vision_model.post_layernorm(patch_pooled)
            patch_pooled = model.visual_projection(patch_pooled)
            patch_pooled_n = F.normalize(patch_pooled + eps, dim=-1)
            seg_image_embeds_n = F.normalize(seg_image_embeds + eps, dim=-1)
            patch_diag = torch.diag(patch_pooled_n @ seg_image_embeds_n.t())
            loss_patch = mse_loss(patch_diag, torch.ones_like(patch_diag))

            # ---- Text-Segment 对齐（保持）----
            text_pooled = torch.cat([
                get_text_tokens_from_segment(org_text_tokens[b:b+1], org_caption[b], seg_caption[b], processor)
                for b in range(bs)
            ], dim=0)
            text_pooled = model.text_model.final_layer_norm(text_pooled)
            text_pooled = model.text_projection(text_pooled)
            text_pooled_n = F.normalize(text_pooled + eps, dim=-1)
            seg_text_embeds_n = F.normalize(seg_text_embeds + eps, dim=-1)
            text_diag = torch.diag(text_pooled_n @ seg_text_embeds_n.t())
            loss_text = mse_loss(text_diag, torch.ones_like(text_diag))

            # ---- Scribble-Text 对齐（保持）----
            has_org_s = has_org_scribble.bool()
            loss_s_t = torch.tensor(0.0, device=fabric.device)
            if has_org_s.any():
                img = F.normalize(org_scribble_embeds[has_org_s] + eps, dim=-1)
                txt = F.normalize(filtered_org_text_embeds[has_org_s] + eps, dim=-1)
                loss_s_t = clip_loss(logit_scale_aux * (img @ txt.t()))

            # ---- RGB-Scribble 一致性（保持）----
            loss_rs = torch.tensor(0.0, device=fabric.device)
            if has_org_s.any():
                loss_rs = F.cosine_embedding_loss(
                    F.normalize(org_image_embeds[has_org_s] + eps, dim=-1),
                    F.normalize(org_scribble_embeds[has_org_s] + eps, dim=-1),
                    torch.ones(has_org_s.sum(), device=fabric.device)
                )

            # ---- 【改造点】Chunk–TopK Segment 多正样本 InfoNCE（T→V） + DEBUG ----
            loss_chunk_align = torch.tensor(0.0, device=fabric.device)
            items_with_chunks = 0

            # 全局边缘库与 mask
            edge_all = F.normalize(edge_seg_embeds_flat + eps, dim=-1)  # [B*K, D]
            valid_all_mask = edge_valid_mask.reshape(-1).to(edge_all.device)  # [B*K]
            neg_inf = torch.finfo(edge_all.dtype).min
            tau = float(max(1e-6, args.chunk_tau))

            # DEBUG 统计容器
            dbg_num_chunks, dbg_num_pos, dbg_pos_ratio, dbg_margin, dbg_entropy = [], [], [], [], []
            with torch.no_grad():
                eff_scale = (model.logit_scale.exp() / tau).item()

            for b in range(bs):
                # 1) 全局对齐滑窗
                caption_for_chunks = filtered_texts_org[b]
                chunk_token_spans = adaptive_sliding_token_spans_global(
                    caption_for_chunks, processor.tokenizer,
                    base_window=args.chunk_base_window, stride=args.chunk_stride, max_spans=64
                )
                if not chunk_token_spans:
                    continue

                text_tokens_b = org_text_tokens[b]
                chunk_embeddings = []
                for s, e in chunk_token_spans:
                    if 0 <= s < e <= text_tokens_b.shape[0]:
                        # 用范数做 soft 权重（比简单 mean 更稳）
                        with torch.no_grad():
                            w = F.softmax(text_tokens_b[s:e].norm(dim=-1), dim=0)  # [L]
                        chunk_embeddings.append((text_tokens_b[s:e] * w[:, None]).sum(dim=0))
                if not chunk_embeddings:
                    continue

                chunk_embeds = torch.stack(chunk_embeddings, dim=0)     # [Nc, H]
                chunk_embeds = model.text_model.final_layer_norm(chunk_embeds)
                chunk_embeds = model.text_projection(chunk_embeds)      # [Nc, D]
                chunk_embeds_n = F.normalize(chunk_embeds, dim=-1)      # [Nc, D]
                Nc = chunk_embeds_n.shape[0]

                # 2) 相似度（限制有效温度，防止过尖）
                scale_eff = (model.logit_scale.exp() / tau).clamp(max=100.0)
                sim_all = scale_eff * (chunk_embeds_n @ edge_all.t())   # [Nc, B*K]

                # 3) 分母（所有有效 edge）
                den_mask = valid_all_mask
                sim_den = sim_all.masked_fill(~den_mask.bool().unsqueeze(0), neg_inf)
                log_den = torch.logsumexp(sim_den, dim=1)               # [Nc]

                # 4) 正样本（本样本的 Top-K 有效 edge）
                pos_mask_row = torch.zeros(bs * K, dtype=torch.bool, device=fabric.device)
                kb = edge_valid_mask[b].to(fabric.device)               # [K]
                if kb.any():
                    idx_start = b * K
                    pos_idx = torch.arange(idx_start, idx_start + K, device=fabric.device)[kb]
                    pos_mask_row[pos_idx] = True
                else:
                    continue

                sim_pos = sim_all.masked_fill(~pos_mask_row.unsqueeze(0), neg_inf)
                log_pos = torch.logsumexp(sim_pos, dim=1)               # [Nc]

                # 5) Multi-Positive InfoNCE
                loss_i = -(log_pos - log_den).mean()
                loss_chunk_align += loss_i
                items_with_chunks += 1

                # ---- DEBUG 统计 ----
                with torch.no_grad():
                    dbg_num_chunks.append(float(Nc))
                    dbg_num_pos.append(float(kb.sum().item()))
                    pos_ratio_i = torch.exp(log_pos - log_den).mean().item()
                    dbg_pos_ratio.append(float(pos_ratio_i))

                    neg_mask_row = den_mask & (~pos_mask_row)
                    if neg_mask_row.any():
                        top_pos = sim_all[:, pos_mask_row].max(dim=1).values
                        top_neg = sim_all[:, neg_mask_row].max(dim=1).values
                        margin_i = (top_pos - top_neg).mean().item()
                        dbg_margin.append(float(margin_i))

                    log_probs = sim_den - log_den.unsqueeze(1)  # [Nc, B*K]
                    probs = torch.exp(log_probs)
                    ent = -(probs * log_probs).sum(dim=1).mean().item()
                    dbg_entropy.append(float(ent))

            if items_with_chunks > 0:
                loss_chunk_align /= items_with_chunks
            else:
                loss_chunk_align = torch.tensor(0.0, device=fabric.device)

            mean_chunks = float(np.mean(dbg_num_chunks)) if dbg_num_chunks else 0.0
            mean_pos = float(np.mean(dbg_num_pos)) if dbg_num_pos else 0.0
            mean_ratio = float(np.mean(dbg_pos_ratio)) if dbg_pos_ratio else 0.0
            mean_margin = float(np.mean(dbg_margin)) if dbg_margin else 0.0
            mean_entropy = float(np.mean(dbg_entropy)) if dbg_entropy else 0.0
            valid_scribble_ratio = float(has_org_s.float().mean().detach().item()) if has_org_s.numel() else 0.0
            valid_chunk_ratio = float(items_with_chunks) / max(1, bs)

            # DEBUG 打印 + wandb
            if (it < 5) or (it % 100 == 0):
                fabric.print(
                    f"[CK DEBUG] it={it} eff_scale={eff_scale:.2f} | "
                    f"Nc(avg)={mean_chunks:.1f} PosK(avg)={mean_pos:.1f} | "
                    f"pos_mass_ratio={mean_ratio:.3f} | top1_margin={mean_margin:.3f} | "
                    f"entropy={mean_entropy:.3f}"
                )
                if wandb.run:
                    wandb.log({
                        "ck/eff_scale": eff_scale,
                        "ck/avg_chunks_per_sample": mean_chunks,
                        "ck/avg_valid_pos_per_sample": mean_pos,
                        "ck/pos_mass_ratio": mean_ratio,
                        "ck/top1_pos_neg_margin": mean_margin,
                        "ck/entropy": mean_entropy,
                    })

            # ---- 损失权重调度 / 自适应反馈 ----
            loss_values_for_weights = {
                "org": loss_org,
                "scribble_text": loss_s_t,
                "rgb_scribble": loss_rs,
                "chunk": loss_chunk_align,
            }
            adaptive_stats = {
                "valid_scribble_ratio": valid_scribble_ratio,
                "valid_chunk_ratio": valid_chunk_ratio,
                "chunk_pos_mass_ratio": mean_ratio,
                "chunk_top1_margin": mean_margin,
            }
            adaptive_weights, adaptive_logs = get_loss_weights(
                args.adaptive_weight_strategy,
                base_loss_weights,
                loss_values_for_weights,
                epoch,
                adaptive_state,
                stats=adaptive_stats,
                alpha=args.adaptive_weight_alpha,
                min_mult=args.adaptive_min_mult,
                max_mult=args.adaptive_max_mult,
                total_epochs=args.epochs,
                temperature=args.adaptive_dwa_temperature,
            )
            lam_s_t_sched = adaptive_weights["scribble_text"]
            lam_rs_sched = adaptive_weights["rgb_scribble"]
            lam_ck_sched = adaptive_weights["chunk"]
            
            # ---- 总损失（两阶段）----
            warmup_sketch_epochs = 0
            if epoch < warmup_sketch_epochs:
                if fabric.global_rank == 0 and i == 0:
                    fabric.print(f"\n🔥🔥🔥 Epoch {epoch} - [Phase 1: Sketch-Only Warmup] 🔥🔥🔥\n")
                loss = (
                    lam_s_t_sched * loss_s_t 
                    + lam_rs_sched * loss_rs
                    + lam_ck_sched * loss_chunk_align
                )
            else:
                if fabric.global_rank == 0 and i == 0:
                    fabric.print(f"\n✅✅✅ Epoch {epoch} - [Phase 2: Joint (Global+Sketch)] ✅✅✅\n")
                joint_training_boost = 1.0
                loss = (
                    joint_training_boost * loss_org
                    # + joint_training_boost * 0.5 * loss_seg
                    # + loss_patch
                    # + loss_text
                    + lam_s_t_sched * loss_s_t
                    + lam_rs_sched * loss_rs
                    + lam_ck_sched * loss_chunk_align
                )

            # ---- 反向传播和优化 ----
            fabric.backward(loss)
            optimizer.step()

            # clamp 一下 logit_scale，避免温度过大
            with torch.no_grad():
                if hasattr(model, "logit_scale"):
                    model.logit_scale.data.clamp_(0, 3.5)

            optimizer.zero_grad()

            # ---- 日志记录（全部 .detach().item()）----
            if fabric.global_rank == 0:
                wandb.log({
                    "iter": it, "lr": lr,
                    "loss_total": loss.detach().item(),
                    "loss_org": loss_org.detach().item(),
                    "loss_seg": loss_seg.detach().item(),
                    "loss_patch": loss_patch.detach().item(),
                    "loss_text": loss_text.detach().item(),
                    "loss_scribble_text": loss_s_t.detach().item(),
                    "loss_rgb_scribble_consistency": loss_rs.detach().item(),
                    "loss_chunk_align": loss_chunk_align.detach().item(),
                    "epoch": epoch,
                    "lam_s_t_eff": float(lam_s_t_sched),
                    "lam_rs_eff": float(lam_rs_sched),
                    "lam_ck_eff": float(lam_ck_sched),
                    **adaptive_logs,
                })
            
            if (it % 10 == 0) or (i == len(train_loader) - 1):
                fabric.print(
                    f"E{epoch} I{it} [{it/total_it*100:.1f}%] "
                    f"lr {lr:.2e} loss {loss.detach().item():.3f} | "
                    f"org {loss_org.detach().item():.3f} seg {loss_seg.detach().item():.3f} "
                    f"patch {loss_patch.detach().item():.3f} txt {loss_text.detach().item():.3f} | "
                    f"ST {loss_s_t.detach().item():.3f} RS {loss_rs.detach().item():.3f} CK {loss_chunk_align.detach().item():.3f} | "
                    f"lam ST {lam_s_t_sched:.4g} RS {lam_rs_sched:.4g} CK {lam_ck_sched:.4g}"
                )
            it += 1
        
        # ---- 保存模型：按存储限制只保留最后一个 checkpoint ----
        fabric.barrier()
        if epoch == args.epochs - 1 and fabric.global_rank == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(sd, save_path)
            fabric.print(f"Final model saved to {save_path}")
        fabric.barrier()

# ========= 参数解析 =========
def get_args_parser():
    parser = argparse.ArgumentParser('CLIP Fine-tuning', add_help=False)
    # Vital
    parser.add_argument('--dataset', required=True, type=str, help='Path to the training JSON file')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to save checkpoints and logs')
    parser.add_argument('--model', default='openai/clip-vit-base-patch16', type=str)
    parser.add_argument('--remove_insect', action='store_true', help='Remove color words in captions')
    parser.add_argument('--remove_colors', action='store_true', help='Remove color words in captions')
    parser.add_argument('--remove_materials', action='store_true', help='Remove material words in captions')
    parser.add_argument('--remove_textures', action='store_true', help='Remove texture words in captions')

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--init_lr', type=float, default=5e-6)
    parser.add_argument('--min_lr', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # Model & Data
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--new_max_token', default=248, type=int)
    
    # Loss Weights
    parser.add_argument('--lambda_scribble_text', type=float, default=0.5, help='Weight for scribble-text alignment loss')
    parser.add_argument('--lambda_rgb_scribble_consistency', type=float, default=0.05, help='Weight for RGB-scribble consistency loss')
    parser.add_argument('--lambda_chunk', type=float, default=0.1, help='Weight for chunk–segment alignment loss')
    parser.add_argument('--adaptive_weight_strategy', type=str, default='fixed',
                        choices=['fixed', 'balance', 'dwa', 'structxlip', 'structxlip_ck_anchor', 'late_structxlip', 'pure_structxlip', 'structxlip_adaptive', 'structxlip_adaptive_v2', 'structxlip_adaptive_v3', 'structxlip_adaptive_v4', 'structxlip_adaptive_v5', 'structxlip_adaptive_v6', 'structxlip_adaptive_v7', 'structxlip_adaptive_v8', 'structxlip_adaptive_v9', 'structxlip_adaptive_v10', 'structxlip_adaptive_v11', 'structxlip_adaptive_v12', 'structxlip_adaptive_v13', 'structxlip_adaptive_v14', 'structxlip_adaptive_v15', 'structxlip_adaptive_v16', 'structxlip_adaptive_v17', 'structxlip_adaptive_v18', 'structxlip_adaptive_v19', 'structxlip_adaptive_v20', 'structxlip_adaptive_v21'],
                        help='Adaptive loss weighting policy. fixed reproduces the original epoch schedule.')
    parser.add_argument('--adaptive_ema_beta', type=float, default=0.95,
                        help='EMA beta used by adaptive weighting policies.')
    parser.add_argument('--adaptive_weight_alpha', type=float, default=0.5,
                        help='Loss-scale balance exponent for adaptive policies.')
    parser.add_argument('--adaptive_min_mult', type=float, default=0.2,
                        help='Minimum multiplier applied around each scheduled base lambda.')
    parser.add_argument('--adaptive_max_mult', type=float, default=5.0,
                        help='Maximum multiplier applied around each scheduled base lambda.')
    parser.add_argument('--adaptive_dwa_temperature', type=float, default=2.0,
                        help='Temperature for DWA adaptive weighting.')

    # Chunk–Segment 对齐相关超参
    parser.add_argument('--chunk_top_k', type=int, default=3, help='每张图取 Top-K segment 作为多正样本')
    parser.add_argument('--chunk_tau', type=float, default=0.07, help='Multi-positive InfoNCE 温度（越小越接近hard）')
    parser.add_argument('--chunk_base_window', type=int, default=3, help='自适应滑窗的基础窗口大小')
    parser.add_argument('--chunk_stride', type=int, default=1, help='滑窗步长')
    parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="Optional custom name for this wandb run.")

    # System
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--wandb_project', type=str, default='CLIP_Finetune_Goal', help='wandb project name')
    
    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
