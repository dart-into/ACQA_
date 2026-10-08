import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



ATTR_NAMES = ["contrast", "brightness", "authenticity", "saturation", "sharpness"]
NUM_ATTRS = 5
SCORE_LEVELS = [0, 1, 2, 3, 4]         
DISTORTION_TYPES = ["contrast", "overexposure", "color_shift", "blur"]  
NUM_DIST_LEVELS = 5                      


@dataclass
class Config:
    # ---- data ----
    data_root: str = "/home/user/data/AQIQA3K"        
    label_csv: str = "/home/user/data/AQIQA3K/label.csv"  
    val_ratio: float = 0.1                             

    # ---- model ----
    vlm_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    clip_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    shared_llm_layers: int = 24          
    clip_feat_dim: int = 512            

    # ---- Stage1: ACAL ----
    acal_lr: float = 1e-6
    acal_epochs: int = 2
    acal_batch_size: int = 2
    n_repeat_score: int = 5              
    w_rr: float = 1.0                    
    w_sr: float = 0.5                    
    w_dr: float = 0.5                    
    acal_baseline_ema: float = 0.9       

    # ---- Stage2: QAL ----
    iqa_lr: float = 1e-4
    iqa_epochs: int = 30
    iqa_batch_size: int = 32
    w_l1: float = 1.0                    
    w_l2: float = 1.0                    
    qal_lr: float = 1e-6
    qal_epochs: int = 2
    w_gr: float = 1.0                   
    w_cr: float = 1.0                    
    w_ar: float = 1.0                    

    # ---- more ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    max_caption_tokens: int = 64


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





class SyntheticDistortion:


    @staticmethod
    def apply(img: torch.Tensor, dist_type: str, level: int) -> torch.Tensor:
        s = level / NUM_DIST_LEVELS  # 0.2 .. 1.0
        if dist_type == "contrast":
            factor = 1.0 - 0.8 * s                    
            return (img - 0.5) * factor + 0.5
        if dist_type == "overexposure":
            return torch.clamp(img + 0.8 * s, 0, 1)   
        if dist_type == "color_shift":
            shift = torch.tensor([0.4 * s, -0.2 * s, 0.1 * s],
                                 device=img.device).view(3, 1, 1)
            return torch.clamp(img + shift, 0, 1)     
        if dist_type == "blur":
            k = 1 + 2 * level                          
            sigma = 0.5 * level
            x = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
            g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            g = (g / g.sum()).view(1, 1, -1)
            img4 = img.unsqueeze(0)
            pad = k // 2
            img4 = F.conv1d(img4.flatten(0, 2).unsqueeze(1),
                            g.expand(img.shape[0], 1, k).flatten(0, 1).view(-1, 1, k),
                            padding=(0, pad),
                            groups=img.shape[0]).view(img.shape[0], img.shape[1], -1, img.shape[2]) \
                if False else img4  
           
            kernel2d = (g.view(1, 1, 1, k) * g.view(1, 1, k, 1)).expand(img.shape[0], 1, k, k)
            blurred = F.conv2d(F.pad(img4, (pad, pad, pad, pad), mode="replicate"),
                               kernel2d, groups=img.shape[0])
            return blurred.squeeze(0)
        raise ValueError(f"unknown distortion type: {dist_type}")




class IQADataset(Dataset):
    

    def __init__(self, samples: List[dict], img_size: int = 224):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(path).convert("RGB")
        tf = T.Compose([T.Resize((self.img_size, self.img_size)), T.ToTensor()])
        return tf(img)  # [C,H,W], [0,1]

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self._load_image(s["path"])
        return {"image": img,
                "mos": torch.tensor(float(s["mos"])),
                "path": s["path"]}


class DistortionRankDataset(Dataset):
  


    def __init__(self, base_images: List[torch.Tensor], img_size: int = 224):
        self.base_images = base_images
        self.img_size = img_size

    def __len__(self):
        return len(self.base_images) * len(DISTORTION_TYPES)

    def __getitem__(self, idx):
        img_idx = idx // len(DISTORTION_TYPES)
        dist_type = DISTORTION_TYPES[idx % len(DISTORTION_TYPES)]
        base = self.base_images[img_idx]
        if base.shape[-1] != self.img_size:
            base = F.interpolate(base.unsqueeze(0), size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False).squeeze(0)
        imgs, levels = [], []
        for lv in range(1, NUM_DIST_LEVELS + 1):
            imgs.append(SyntheticDistortion.apply(base, dist_type, lv))
            levels.append(lv)
        return {"images": torch.stack(imgs), "levels": torch.tensor(levels),
                "dist_type": dist_type}





class DualBranchVLM(nn.Module):
   



    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(cfg.vlm_name)
        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.vlm_name, torch_dtype=torch.float32)

        self.visual = full.visual                       
        base_layers = full.model.layers                 
        n = cfg.shared_llm_layers
        self.shared_layers = nn.ModuleList(base_layers[:n])          
        self.content_layers = nn.ModuleList(
            [copy.deepcopy(l) for l in base_layers[n:]])           
        self.attr_layers = nn.ModuleList(
            [copy.deepcopy(l) for l in base_layers[n:]])           
        self.content_head = copy.deepcopy(full.lm_head)
        self.attr_head = copy.deepcopy(full.lm_head)
        self.norm_c = copy.deepcopy(full.model.norm)
        self.norm_a = copy.deepcopy(full.model.norm)

        # [FIX P0-6b] `_forward_branch` needs helpers that only exist on the
        # pretrained objects: the 3D M-RoPE index (`get_rope_index`, defined on
        # the CausalLM) and the 4D causal mask (`_update_causal_mask`, defined
        # on the inner model). Keep them, and reuse the base embedding matrix
        # directly instead of loading a *second* full copy of the 7B weights
        # later in main(). The duplicated tail layers [n:] are dead weight here
        # (we hold deep copies) and are released by emptying the list.
        self.embed_tokens = full.model.embed_tokens
        self.base_model = full.model
        self.base_model.layers = nn.ModuleList()
        # stored as a bound method (not a Module) so it is not registered as a
        # submodule and does not pollute state_dict() / parameters()
        self._get_rope_index = full.get_rope_index
        del full

       
        self.score_token_ids = torch.tensor(
            [self.processor.tokenizer.encode(str(s), add_special_tokens=False)[-1]
             for s in SCORE_LEVELS])

        
        self.attr_prompts = {
            "contrast":    "Rate the CONTRAST quality of this image with an integer score from 0 to 4 (0 worst, 4 best). Answer with a single digit.",
            "brightness":  "Rate the BRIGHTNESS quality of this image with an integer score from 0 to 4 (0 worst, 4 best). Answer with a single digit.",
            "authenticity":"Rate the AUTHENTICITY (freedom from color/artifact distortion) of this image with an integer score from 0 to 4 (0 worst, 4 best). Answer with a single digit.",
            "saturation":  "Rate the color SATURATION quality of this image with an integer score from 0 to 4 (0 worst, 4 best). Answer with a single digit.",
            "sharpness":   "Rate the SHARPNESS of this image with an integer score from 0 to 4 (0 worst, 4 best). Answer with a single digit.",
        }
        self.caption_prompt = "Describe the content of this image in one short sentence."

    

    def _embed_inputs(self, pil_images, prompts: List[str]):
        
        conversations = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": p}]}]
            for img, p in zip(pil_images, prompts)
        ]
        texts = [self.processor.apply_chat_template(c, tokenize=False,
                                                    add_generation_prompt=True)
                 for c in conversations]
        inputs = self.processor(text=texts, images=pil_images,
                                return_tensors="pt", padding=True)
        return {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

    def _forward_branch(self, inputs, branch: str, logits_needed: bool = True):
        """Run the shared backbone, then the content or the attribute branch.

        [FIX P0-6b] The original implementation called every decoder layer with
        only the raw 2D padding mask and ``position_ids`` taken from the
        processor output. That cannot work with Qwen2.5-VL on
        ``transformers>=4.49``:

        * the processor does not emit ``position_ids`` (they are 3D M-RoPE
          indices produced by ``get_rope_index``), so ``pos`` was ``None``;
        * ``Qwen2_5_VLDecoderLayer`` requires ``position_embeddings`` (cos, sin).
          Left as ``None`` the attention module raises ``AttributeError:
          'NoneType' object has no attribute 'unsqueeze'``;
        * SDPA expects the 4D additive causal mask built by
          ``_update_causal_mask``, not the 2D padding mask.

        All three are now prepared exactly as ``Qwen2_5_VLModel.forward`` does.
        """
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        attention_mask = inputs.get("attention_mask")

        emb = self._token_embed(input_ids)
        if pixel_values is not None:
            vis = self.visual(pixel_values, grid_thw=image_grid_thw)
            mask = (input_ids == self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
            if mask.any():
                emb = emb.clone()
                emb[mask] = vis.to(emb.dtype)

        seq_len = emb.shape[1]
        device = emb.device

        # 3D M-RoPE position ids: [3, B, T]
        pos = inputs.get("position_ids")
        if pos is None:
            pos, _ = self._get_rope_index(
                input_ids, image_grid_thw, None,
                inputs.get("second_per_grid_ts"), attention_mask)

        cache_position = torch.arange(seq_len, device=device)
        causal_mask = self.base_model._update_causal_mask(
            attention_mask, emb, cache_position, None, False)
        position_embeddings = self.base_model.rotary_emb(emb, pos)

        layer_kwargs = dict(attention_mask=causal_mask,
                            position_ids=pos,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings)

        h = emb
        for layer in self.shared_layers:
            h = layer(h, **layer_kwargs)[0]

        layers = self.content_layers if branch == "content" else self.attr_layers
        norm = self.norm_c if branch == "content" else self.norm_a
        head = self.content_head if branch == "content" else self.attr_head
        for layer in layers:
            h = layer(h, **layer_kwargs)[0]
        h = norm(h)
        logits = head(h) if logits_needed else None
        return logits, h

    def _token_embed(self, input_ids):
        
        if not hasattr(self, "embed_tokens"):
            raise RuntimeError("construct after completing vlm.set_embeddings(model.model.embed_tokens)")
        return self.embed_tokens(input_ids)

    def set_embeddings(self, embed_tokens: nn.Module):
        self.embed_tokens = embed_tokens

    
    @torch.no_grad()
    def generate_caption(self, pil_images) -> List[str]:
       
        prompts = [self.caption_prompt] * len(pil_images)
        inputs = self._embed_inputs(pil_images, prompts)
        
        logits, _ = self._forward_branch(inputs, "content")
       
        return self._greedy_decode_simple(inputs, logits)

    def _greedy_decode_simple(self, inputs, first_logits) -> List[str]:
        
        nxt = first_logits[:, -1, :].argmax(-1)
        return [self.processor.tokenizer.decode([t]) for t in nxt]

    def score_attribute_logits(self, pil_image, attr: str) -> torch.Tensor:
       
        inputs = self._embed_inputs([pil_image], [self.attr_prompts[attr]])
        logits, _ = self._forward_branch(inputs, "attr")
        last = logits[0, -1, :]                       
        logp = F.log_softmax(last, dim=-1)
        return logp[self.score_token_ids.to(logp.device)]   

    def score_all_attributes_expected(self, pil_image,
                                      differentiable: bool = False) -> torch.Tensor:
       
        levels = torch.tensor(SCORE_LEVELS, dtype=torch.float32,
                              device=next(self.parameters()).device)
        out = []
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with ctx:
            for attr in ATTR_NAMES:
                logp = self.score_attribute_logits(pil_image, attr)
                p = F.softmax(logp, dim=-1)
                out.append((p * levels).sum())
        return torch.stack(out)

    def sample_attribute_scores(self, pil_image, temperature: float = 1.0
                                ) -> Tuple[List[int], torch.Tensor]:
       
        sampled, logps = [], 0.0
        for attr in ATTR_NAMES:
            logp = self.score_attribute_logits(pil_image, attr)
            dist = torch.distributions.Categorical(logits=logp / temperature)
            idx = dist.sample()
            sampled.append(SCORE_LEVELS[idx.item()])
            logps = logps + dist.log_prob(idx)
        return sampled, logps


# Stage 2   CLIP-IQA 

class CLIPIQAModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        import open_clip
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip_name, pretrained=cfg.clip_pretrained)
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_name)
        self.visual = clip_model.visual
        self.text_encoder = clip_model
        
        for p in self.visual.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.head = nn.Linear(2 * cfg.clip_feat_dim, NUM_ATTRS + 1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)                       # f_I

    @property
    def _device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def encode_text(self, captions: List[str]) -> torch.Tensor:
        # [FIX P0-6] `open_clip.get_tokenizer` returns CPU tensors while the
        # encoders live on cfg.device -> RuntimeError without this move.
        toks = self.tokenizer(captions).to(self._device)
        return self.text_encoder.encode_text(toks)       # f_T

    def forward(self, images: torch.Tensor,
                captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_i = self.encode_image(images)
        f_t = self.encode_text(captions)
        fused = self.head(torch.cat([f_i, f_t], dim=-1))  # [B,6]
        attr_pred = fused[:, :NUM_ATTRS]                  
        quality_pred = fused[:, NUM_ATTRS]                
        return attr_pred, quality_pred, (f_i, f_t)



def srcc(pred: np.ndarray, label: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(pred, label).correlation)


def plcc(pred: np.ndarray, label: np.ndarray) -> float:
    from scipy.stats import pearsonr
    return float(pearsonr(pred, label)[0])


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
# [FIX P0-5] Every submodule that Stage 2 puts into an optimizer is now saved.
# The previous `torch.save` wrote only `attr_layers`, `content_layers` and
# `iqa.head`, silently discarding `attr_head` / `norm_a` / `content_head` /
# `norm_c` -- i.e. four of the six trained tensors groups -- so a loaded
# checkpoint could not reproduce the reported numbers.
CKPT_MODULES = {
    "vlm_attr":         lambda vlm: vlm.attr_layers,
    "vlm_attr_head":    lambda vlm: vlm.attr_head,
    "vlm_attr_norm":    lambda vlm: vlm.norm_a,
    "vlm_content":      lambda vlm: vlm.content_layers,
    "vlm_content_head": lambda vlm: vlm.content_head,
    "vlm_content_norm": lambda vlm: vlm.norm_c,
}


def save_checkpoint(path: str, vlm: "DualBranchVLM", iqa: "CLIPIQAModel",
                    cfg: Optional[Config] = None,
                    metrics: Optional[dict] = None) -> dict:
    """Persist the trained deltas of ACQA (the shared backbone stays on HF)."""
    state = {
        key: {k: t.detach().cpu() for k, t in getter(vlm).state_dict().items()}
        for key, getter in CKPT_MODULES.items()
    }
    state["iqa_head"] = {k: t.detach().cpu()
                         for k, t in iqa.head.state_dict().items()}
    if cfg is not None:
        from dataclasses import asdict
        state["config"] = asdict(cfg)
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)
    return state


def load_checkpoint(path: str, vlm: "DualBranchVLM", iqa: "CLIPIQAModel",
                    strict: bool = True, map_location: str = "cpu") -> dict:
    """Inverse of :func:`save_checkpoint`. Build the model first, then call."""
    ck = torch.load(path, map_location=map_location)
    for key, getter in CKPT_MODULES.items():
        if key in ck:
            getter(vlm).load_state_dict(ck[key], strict=strict)
    if "iqa_head" in ck:
        iqa.head.load_state_dict(ck["iqa_head"], strict=strict)
    return ck


# Stage 1:  calibrate ACAL

class AttributeCalibrationRL:


    def __init__(self, vlm: DualBranchVLM, cfg: Config):
        self.vlm = vlm
        self.cfg = cfg
        params = (list(vlm.attr_layers.parameters()) +
                  list(vlm.attr_head.parameters()) +
                  list(vlm.norm_a.parameters()))
        self.optimizer = torch.optim.AdamW(params, lr=cfg.acal_lr)
        self.baseline = 0.0   


    def rank_reward(self, pil_images_level: List, true_levels: List[int],
                    dist_type: str) -> float:


        rel = {"contrast": ["contrast"],
               "overexposure": ["brightness"],
               "color_shift": ["authenticity", "saturation"],
               "blur": ["sharpness"]}[dist_type]

        scores = []
        with torch.no_grad():
            for img in pil_images_level:
                s = 0.0
                for a in rel:
                    logp = self.vlm.score_attribute_logits(img, a)
                    p = F.softmax(logp, dim=-1)
                    levels = torch.tensor(SCORE_LEVELS, dtype=torch.float32,
                                          device=p.device)
                    s += float((p * levels).sum())
                scores.append(s / len(rel))
        scores = np.array(scores)

        
        true = np.array([-(lv - 1) for lv in true_levels], dtype=np.float64)
        p_true = np.exp(true - true.max()); p_true /= p_true.sum()
        p_pred = np.exp(scores - scores.max()); p_pred /= p_pred.sum()
        p_pred = np.clip(p_pred, 1e-8, 1.0)

        kl = float(np.sum(p_true * np.log(p_true / p_pred)))  # KL(true || pred)
        return math.exp(-kl)                                  


    def stability_reward(self, pil_image) -> float:
        
        var_sum = 0.0
        for _ in range(self.cfg.n_repeat_score):
            sampled, _ = self.vlm.sample_attribute_scores(pil_image, temperature=1.0)
            var_sum += float(np.var(sampled))
        avg_var = var_sum / self.cfg.n_repeat_score
        return math.exp(-avg_var)              

    
    def disentanglement_reward(self, pil_images, target_attr: str,
                               attr_snapshot: Dict[str, torch.Tensor]) -> float:
        
        others = [a for a in ATTR_NAMES if a != target_attr]
        diff = 0.0
        with torch.no_grad():
            for img in pil_images:
                for a in others:
                    logp = self.vlm.score_attribute_logits(img, a)
                    p = F.softmax(logp, dim=-1)
                    levels = torch.tensor(SCORE_LEVELS, dtype=torch.float32,
                                          device=p.device)
                    cur = (p * levels).sum()
                    diff += float((cur - attr_snapshot[a]).abs())
        avg_diff = diff / max(1, len(pil_images) * len(others))
        return math.exp(-avg_diff)

    
    def train(self, rank_loader: DataLoader, eval_images: List,
              pil_fn, epochs: Optional[int] = None):
        # NOTE: `eval_images` is currently unused; it is kept for API
        # compatibility and is intended for periodic hold-out probing.
        epochs = epochs or self.cfg.acal_epochs
        # [FIX P0-3c] the clip list must match the optimizer's parameter set,
        # otherwise `norm_a` is stepped without ever being clipped.
        attr_params = (list(self.vlm.attr_layers.parameters()) +
                       list(self.vlm.attr_head.parameters()) +
                       list(self.vlm.norm_a.parameters()))

        for ep in range(epochs):
            ep_reward, ep_loss, n_samples, n_batches = 0.0, 0.0, 0, 0
            for batch in rank_loader:
                imgs = batch["images"]            # [B,5,C,H,W]
                levels = batch["levels"]          # [B,5]
                dist_types = batch["dist_type"]   # list[str]
                B = imgs.shape[0]

                # [FIX P0-3] `policy_term` accumulates advantage * log pi(a|s).
                # The previous version summed the raw log-probs and dropped the
                # advantage entirely, so the gradient carried no reward signal
                # at all (equivalent to unconditionally maximising the sampled
                # tokens' log-prob).
                policy_term = 0.0
                batch_reward = 0.0
                deferred_probe = None
                for b in range(B):
                    pil_levels = [pil_fn(imgs[b, i]) for i in range(imgs.shape[1])]

                    # --- R_rr：images of 5 levels dist log-prob ---
                    dt = dist_types[b]
                    r_rr = self.rank_reward(pil_levels,
                                            levels[b].tolist(), dt)

                    
                    _, logp_sample = self.vlm.sample_attribute_scores(pil_levels[0])

                    # --- R_sr ---
                    r_sr = self.stability_reward(pil_levels[0])

                    # --- R_dr ---
                    snapshot = {}
                    with torch.no_grad():
                        for a in ATTR_NAMES:
                            logp_a = self.vlm.score_attribute_logits(pil_levels[0], a)
                            p_a = F.softmax(logp_a, -1)
                            lv = torch.tensor(SCORE_LEVELS, dtype=torch.float32,
                                              device=p_a.device)
                            snapshot[a] = (p_a * lv).sum()
                    rel_attr = {"contrast": "contrast",
                                "overexposure": "brightness",
                                "color_shift": "saturation",
                                "blur": "sharpness"}[dt]
                    # [FIX P0-3d] the entropy-descent probe mutates `attr_head`
                    # in place. It used to run here, i.e. *after* `logp_sample`
                    # had already been recorded but *before* `loss.backward()`,
                    # which trips autograd's version check ("a variable needed
                    # for gradient computation has been modified by an inplace
                    # operation"). R_dr is still measured here; only the weight
                    # mutation is deferred to after `optimizer.step()`, which
                    # also matches Algorithm 1 (update theta_att, *then* probe).
                    r_dr = self.disentanglement_reward(pil_levels[:1], rel_attr,
                                                       snapshot)
                    deferred_probe = (pil_levels[0], rel_attr)

                    reward = (self.cfg.w_rr * r_rr + self.cfg.w_sr * r_sr +
                              self.cfg.w_dr * r_dr)
                    advantage = reward - self.baseline
                    self.baseline = (self.cfg.acal_baseline_ema * self.baseline +
                                     (1 - self.cfg.acal_baseline_ema) * reward)
                    policy_term = policy_term + advantage * logp_sample
                    batch_reward += reward

                # REINFORCE loss: -(advantage * log pi). TODO(paper Eq. 7-8):
                # the manuscript specifies a *clipped PPO* objective with a KL
                # regulariser and group-wise advantage normalisation; this is
                # still plain REINFORCE with an EMA baseline.
                loss = -(policy_term / max(1, B))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(attr_params, 1.0)
                self.optimizer.step()

                # [FIX P0-3d] now safe: the backward graph has been freed.
                if deferred_probe is not None:
                    self._single_attr_probe_update(*deferred_probe)

                ep_reward += batch_reward
                ep_loss += float(loss.detach())
                n_samples += B
                n_batches += 1

            print(f"[ACAL] epoch {ep+1}/{epochs}  "
                  f"avg_reward={ep_reward/max(1,n_samples):.4f}  "
                  f"loss={ep_loss/max(1,n_batches):.4f}  "
                  f"baseline={self.baseline:.4f}")

    def _single_attr_probe_update(self, pil_image, attr: str):
        # [FIX P0-3b] This was decorated with @torch.no_grad(), which stops the
        # graph from being built, so the `torch.autograd.grad(entropy, ...)`
        # below raised "element 0 of tensors does not require grad" and Stage 1
        # could never reach its second batch. The decorator is removed; the
        # in-place parameter step is still guarded by its own no_grad block.
        #
        # TODO(paper): this entropy-descent probe on `attr_head` is not
        # described anywhere in the manuscript. Either formalise it (it belongs
        # next to R_dr in Sec. 3.2) or drop it.
        logp = self.vlm.score_attribute_logits(pil_image, attr)
        p = F.softmax(logp, dim=-1)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        grads = torch.autograd.grad(entropy, [q for q in self.vlm.attr_head.parameters()
                                              if q.requires_grad],
                                    retain_graph=False, allow_unused=True)
        with torch.no_grad():
            lr = self.cfg.acal_lr
            for q, g in zip([q for q in self.vlm.attr_head.parameters() if q.requires_grad],
                            grads):
                if g is not None:
                    q -= lr * g



# Stage 2: quality-aware learning QAL

class QualityAwareLearning:
    

    def __init__(self, vlm: DualBranchVLM, cfg: Config,
                 iqa: Optional[CLIPIQAModel] = None):
        self.vlm = vlm
        self.cfg = cfg
        # [FIX P0-4b] An already-trained IQA model can be injected, so that
        # evaluation can never silently run on a freshly-initialised head.
        self.iqa = (iqa if iqa is not None else CLIPIQAModel(cfg)).to(cfg.device)
        self.iqa_optimizer = torch.optim.AdamW(
            self.iqa.head.parameters(), lr=cfg.iqa_lr)
        self.content_optimizer = torch.optim.AdamW(
            list(vlm.content_layers.parameters()) +
            list(vlm.content_head.parameters()) +
            list(vlm.norm_c.parameters()), lr=cfg.qal_lr)
        self.attr_optimizer = torch.optim.AdamW(
            list(vlm.attr_layers.parameters()) +
            list(vlm.attr_head.parameters()) +
            list(vlm.norm_a.parameters()), lr=cfg.qal_lr)

   
    def _clip_normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    # ---- R_gr ----
    def content_alignment_step(self, loader: DataLoader, pil_fn):
        
        self.iqa.eval()
        epoch_reward = 0.0
        n = 0
        for batch in loader:
            images = batch["image"].to(self.cfg.device)
            pil_images = [pil_fn(im) for im in images]
            captions = self.vlm.generate_caption(pil_images)

            with torch.no_grad():
                f_i = self.iqa.encode_image(self._clip_normalize(images))
                f_t = self.iqa.encode_text(captions)
                r_gr = F.cosine_similarity(f_i, f_t, dim=-1)   # [B]

            # REINFORCE：max alignment of caption and images
            inputs = self.vlm._embed_inputs(pil_images,
                                            [self.vlm.caption_prompt] * len(pil_images))
            logits, _ = self.vlm._forward_branch(inputs, "content")
            
            logp = self._caption_logprob(logits, inputs, captions)
            loss = -((r_gr.mean() - 0.0) * logp)  

            self.content_optimizer.zero_grad()
            loss.backward()
            self.content_optimizer.step()

            epoch_reward += float(r_gr.mean()); n += 1
        print(f"[QAL/R_gr]  alignment  reward = {epoch_reward/max(1,n):.4f}")

    def _caption_logprob(self, logits, inputs, captions) -> torch.Tensor:
        
        ids = self.vlm.processor.tokenizer(
            captions, add_special_tokens=False)["input_ids"]
        logp_all = F.log_softmax(logits[:, -1, :], dim=-1)
        tot = 0.0
        for b, toks in enumerate(ids):
            if len(toks) > 0:
                tot = tot + logp_all[b, toks[0]]
        return tot / max(1, len(ids))

    # ---- Step B: （L_1 + L_2）----
    def train_iqa(self, loader: DataLoader, captions_cache: Optional[Dict] = None,
                  epochs: Optional[int] = None) -> CLIPIQAModel:
       
        epochs = epochs or self.cfg.iqa_epochs
        self.iqa.train()
        for ep in range(epochs):
            total = 0.0
            for batch in loader:
                images = batch["image"].to(self.cfg.device)
                mos = batch["mos"].to(self.cfg.device)
                pil_images = [self._to_pil(im) for im in images]

                
                with torch.no_grad():
                    attr_target = torch.stack(
                        [self.vlm.score_all_attributes_expected(p)
                         for p in pil_images]).to(self.cfg.device)  # [B,5] (0-4)

                
                if captions_cache is not None:
                    captions = [captions_cache[id(b)] for b in batch["image"]]
                else:
                    captions = self.vlm.generate_caption(pil_images)

                attr_pred, q_pred, _ = self.iqa(
                    self._clip_normalize(images), captions)

                l1 = F.mse_loss(attr_pred, attr_target)       
                l2 = F.mse_loss(q_pred, mos * 4.0)            
                loss = self.cfg.w_l1 * l1 + self.cfg.w_l2 * l2

                self.iqa_optimizer.zero_grad()
                loss.backward()
                self.iqa_optimizer.step()
                total += float(loss)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[IQA] epoch {ep+1}/{epochs}  loss={total/max(1,len(loader)):.4f}")
        return self.iqa

    def _to_pil(self, img_tensor: torch.Tensor):
        from PIL import Image
        arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
               ).astype(np.uint8)
        return Image.fromarray(arr)

    # ---- Step C: R_cr ----
    def attribute_contribution_reward(self, loader: DataLoader) -> torch.Tensor:
        
        self.iqa.eval()
        rewards = torch.zeros(NUM_ATTRS)
        n = 0
        for batch in loader:
            images = batch["image"].to(self.cfg.device)
            mos = batch["mos"].to(self.cfg.device)
            captions = self.vlm.generate_caption([self._to_pil(im) for im in images])

            f_i = self.iqa.encode_image(self._clip_normalize(images))
            f_t = self.iqa.encode_text(captions)
            z = torch.cat([f_i, f_t], dim=-1).detach().requires_grad_(True)
            fused = self.iqa.head(z)                 # [B,6]

            # gradient of the quality output w.r.t. the fused feature
            g_q, = torch.autograd.grad(fused[:, NUM_ATTRS].sum(), z,
                                       retain_graph=True)
            for k in range(NUM_ATTRS):
                g_k, = torch.autograd.grad(fused[:, k].sum(), z,
                                           retain_graph=(k < NUM_ATTRS - 1))
                cos = F.cosine_similarity(
                    g_k.flatten(1), g_q.flatten(1), dim=-1).mean().cpu()
                rewards[k] += max(0.0, float(cos))   
            n += 1
        return rewards / max(1, n)

    # ---- Step D:  R_ar ----
    def attribute_gain_reward(self, train_loader: DataLoader,
                              val_loader: DataLoader,
                              srcc_before: float, plcc_before: float
                              ) -> float:
       
        self.train_iqa(train_loader, epochs=max(5, self.cfg.iqa_epochs // 3))
        s_after, p_after = self.evaluate_iqa(val_loader)
        gain = 0.5 * (s_after - srcc_before) + 0.5 * (p_after - plcc_before)
        print(f"[QAL/R_ar] SRCC {srcc_before:.4f}->{s_after:.4f}  "
              f"PLCC {plcc_before:.4f}->{p_after:.4f}  gain={gain:+.4f}")
        return gain

    @torch.no_grad()
    def evaluate_iqa(self, loader: DataLoader) -> Tuple[float, float]:
        self.iqa.eval()
        preds, labels = [], []
        for batch in loader:
            images = batch["image"].to(self.cfg.device)
            captions = self.vlm.generate_caption([self._to_pil(im) for im in images])
            _, q_pred, _ = self.iqa(self._clip_normalize(images), captions)
            preds.append(q_pred.cpu().numpy())
            labels.append(batch["mos"].numpy() * 4.0)
        preds = np.concatenate(preds); labels = np.concatenate(labels)
        return srcc(preds, labels), plcc(preds, labels)

    def run(self, train_loader: DataLoader, val_loader: DataLoader,
            pil_fn=None):
        pil_fn = pil_fn or self._to_pil

        
        self.content_alignment_step(train_loader, pil_fn)

        
        self.train_iqa(train_loader)
        srcc0, plcc0 = self.evaluate_iqa(val_loader)
        print(f"[QAL] IQA converged: SRCC={srcc0:.4f}  PLCC={plcc0:.4f}")

        
        r_cr = self.attribute_contribution_reward(train_loader)
        self._update_attr_branch_with_reward(r_cr.mean().item(),
                                             train_loader, self.cfg.w_cr)

       
        gain = self.attribute_gain_reward(train_loader, val_loader, srcc0, plcc0)
        self._update_attr_branch_with_reward(gain, train_loader, self.cfg.w_ar)

        return self.iqa

    def _update_attr_branch_with_reward(self, reward: float,
                                        loader: DataLoader, weight: float):
        
        adv = weight * reward
        logp_total = 0.0
        cnt = 0
        for batch in loader:
            for im in batch["image"]:
                _, logp = self.vlm.sample_attribute_scores(self._to_pil(im))
                logp_total = logp_total + logp
                cnt += 1
                if cnt >= 8:      
                    break
            if cnt >= 8:
                break
        loss = -(adv * logp_total / max(1, cnt))
        self.attr_optimizer.zero_grad()
        loss.backward()
        self.attr_optimizer.step()
        print(f"[QAL] attri REINFORCE update: reward={reward:.4f} loss={float(loss):.4f}")



class FinalQualityScorer:
    
    def __init__(self, vlm: DualBranchVLM, iqa: CLIPIQAModel, cfg: Config):
        self.vlm = vlm
        self.iqa = iqa
        self.cfg = cfg

    def retrain_final(self, train_loader: DataLoader, epochs: int = 30):
        # [FIX P0-4a] Returns `self`, not the helper object. The old version
        # made main() rebind `scorer` to a QualityAwareLearning, which has no
        # .score() -> AttributeError on the very next call.
        qal = QualityAwareLearning(self.vlm, self.cfg, iqa=self.iqa)
        qal.train_iqa(train_loader, epochs=epochs)
        return self

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """SRCC / PLCC of the *trained* IQA head (never a fresh one)."""
        return QualityAwareLearning(self.vlm, self.cfg,
                                    iqa=self.iqa).evaluate_iqa(loader)

    @torch.no_grad()
    def score(self, pil_images: List) -> Dict[str, np.ndarray]:
        
        self.iqa.eval()
        captions = self.vlm.generate_caption(pil_images)
        attrs = np.stack([
            self.vlm.score_all_attributes_expected(p).cpu().numpy()
            for p in pil_images])
        images = torch.stack([
            self._preprocess(p) for p in pil_images]).to(self.cfg.device)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=images.device).view(1, 3, 1, 1)
        _, q_pred, _ = self.iqa((images - mean) / std, captions)
        return {"quality": q_pred.cpu().numpy(),
                "attributes": attrs,           # [N,5]，0-4
                "captions": captions}

    def _preprocess(self, pil_img) -> torch.Tensor:
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        return tf(pil_img.convert("RGB"))



def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    iqa_samples = load_iqa_dataset(cfg)

    def to_pil(img_tensor):
        from PIL import Image
        arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
               ).astype(np.uint8)
        return Image.fromarray(arr)

    def load_image_tensor(path: str) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(path).convert("RGB")
        return T.Compose([T.Resize((224, 224)), T.ToTensor()])(img)

    
    rng = random.Random(cfg.seed)
    idx_all = list(range(len(iqa_samples)))
    rng.shuffle(idx_all)
    n_val = max(1, int(len(idx_all) * cfg.val_ratio))
    val_samples = [iqa_samples[i] for i in idx_all[:n_val]]
    train_samples = [iqa_samples[i] for i in idx_all[n_val:]]
    print(f"[data] train {len(train_samples)}   val {len(val_samples)}")

    train_set = IQADataset(train_samples)
    val_set = IQADataset(val_samples)
    train_loader = DataLoader(train_set, batch_size=cfg.iqa_batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=cfg.iqa_batch_size,
                            shuffle=False, num_workers=2)

    
    base_images = [load_image_tensor(s["path"]) for s in train_samples[:64]]
    rank_set = DistortionRankDataset(base_images)
    rank_loader = DataLoader(rank_set, batch_size=1, shuffle=True)

    
    print("== Stage 0: dual branch VLM ==")
    vlm = DualBranchVLM(cfg).to(device)
    # [FIX P0-6b / P1-10] DualBranchVLM now retains the base model's embedding
    # matrix itself, so the second full `from_pretrained` -- an extra ~28 GB of
    # fp32 weights held live at peak -- is gone. `set_embeddings()` is kept only
    # for backwards compatibility with externally built models.

    
    print("== Stage 1: ACAL  ==")
    acal = AttributeCalibrationRL(vlm, cfg)
    acal.train(rank_loader, [to_pil(im) for im in base_images[:8]], to_pil)

    
    print("== Stage 2: QAL  ==")
    qal = QualityAwareLearning(vlm, cfg)
    iqa = qal.run(train_loader, val_loader, pil_fn=to_pil)

    print("== Stage 3: test quality score ==")
    scorer = FinalQualityScorer(vlm, iqa, cfg)
    scorer.retrain_final(train_loader, epochs=cfg.iqa_epochs)   # [FIX P0-4a]
    s_final, p_final = scorer.evaluate(val_loader)              # [FIX P0-4b]
    print(f"final model: SRCC={s_final:.4f}  PLCC={p_final:.4f}")

    
    demo_images = [load_image_tensor(s["path"]).permute(1, 2, 0).numpy()
                   for s in val_samples[:4]]
    from PIL import Image
    demo_images = [Image.fromarray((im * 255).astype(np.uint8))
                   for im in demo_images]
    results = scorer.score(demo_images)
    for i, cap in enumerate(results["captions"]):
        print(f"  image  {i}: quality={results['quality'][i]:.3f} "
              f"attrs={np.round(results['attributes'][i],2).tolist()} | {cap}")

    
    save_checkpoint("acqa_checkpoint.pt", vlm, iqa, cfg,
                    metrics={"srcc": s_final, "plcc": p_final})
    print("saved acqa_checkpoint.pt "
          f"({len(CKPT_MODULES) + 1} module groups + config + metrics)")


def load_iqa_dataset(cfg: "Config") -> List[dict]:
    
    import csv
    import os

    records = []
    with open(cfg.label_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        if header is not None:
            try:
                float(header[1])
                rows = [header] + list(reader)
            except (ValueError, IndexError):
                rows = list(reader)
        else:
            rows = []
        for row in rows:
            if len(row) < 2 or not row[0].strip():
                continue
            fname = row[0].strip()
            path = fname if os.path.isabs(fname) else os.path.join(cfg.data_root, fname)
            if not os.path.isfile(path):
                print(f"[data] missing image, skipped -> {path}")
                continue
            records.append({"path": path, "raw_mos": float(row[1])})

    if not records:
        raise RuntimeError(
            f"no usable samples found: checked {cfg.label_csv} "
            f"against image root {cfg.data_root}")


    mos_all = np.array([r["raw_mos"] for r in records], dtype=np.float64)
    lo, hi = float(mos_all.min()), float(mos_all.max())
    rng = hi - lo if hi > lo else 1.0
    for r in records:
        r["mos"] = (r.pop("raw_mos") - lo) / rng

    print(f"[data] loaded {len(records)} images from {cfg.label_csv} "
          f"(MOS min-max normalized to [0, 1], raw range [{lo:.3f}, {hi:.3f}])")
    return records


if __name__ == "__main__":
    main()
