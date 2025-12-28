import os
import sys
import warnings
from pathlib import Path
from functools import partial
from collections import defaultdict

import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import LOGGER  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402


class T2IEvaluator:
    def __init__(self, anno_path, save_dir, device, max_seq_len=None, n_samples=None):
        self.device = device
        self.save_dir = save_dir
        self.n_samples = n_samples
        self.cate, self.cate2prmt = self._prepare_prompts(
            anno_path=anno_path, max_seq_len=max_seq_len
        )
        self.origin_path = self.save_dir / "origin_out"
        self.compress_path = self.save_dir / "compress_out"

    def _prepare_prompts(self, anno_path, max_seq_len):
        annos = json.load(open(anno_path, "r"))
        cate2prmt = defaultdict(list)
        for v in annos.values():
            cate2prmt[v["category"]].append(v["prompt"])
        cate = [*cate2prmt.keys()]

        if max_seq_len is not None:

            def is_valid(n, max_len):
                return True if len(n) <= max_len else False

            for c in cate:
                cate2prmt[c] = list(
                    filter(partial(is_valid, max_len=max_seq_len), cate2prmt[c])
                )

        if self.n_samples is not None:
            for c in cate:
                cate2prmt[c] = cate2prmt[c][: self.n_samples]

        sampled_cate2prmt = defaultdict()
        for c in cate:
            x = {}
            for i, v in enumerate(cate2prmt[c]):
                x[f"{i:05d}"] = v
            sampled_cate2prmt[c] = x

        with open(str(self.save_dir / "eval_prompts.json"), "w") as f:
            json.dump(sampled_cate2prmt, f)

        return cate, sampled_cate2prmt

    def collect_origin_out(self, model, num_inference_steps):
        LOGGER.info("Collecting output from original model...")

        model = model.to(self.device)
        pg_bar = tqdm(range(len(self.cate)))
        for i in pg_bar:
            c = self.cate[i]
            pg_bar.set_description(f"Category[{c}]")
            dst_path = self.origin_path / c
            os.makedirs(dst_path, exist_ok=True)

            prompts = [*self.cate2prmt[c].values()]
            outs = model.generate_image(
                prompts, num_inference_steps=num_inference_steps
            )
            for i in range(len(outs)):
                outs[i].save(dst_path / f"{i:05d}.png")

        cleanup_memory(verbose=True)

    def collect_compress_out(self, model, num_inference_steps):
        LOGGER.info("Collecting output from compressed model...")

        model = model.to(self.device)
        pg_bar = tqdm(range(len(self.cate)))
        for i in pg_bar:
            c = self.cate[i]
            pg_bar.set_description(f"Category[{c}]")
            dst_path = self.compress_path / c
            os.makedirs(dst_path, exist_ok=True)

            prompts = [*self.cate2prmt[c].values()]
            outs = model.generate_image(
                prompts, num_inference_steps=num_inference_steps
            )
            for i in range(len(outs)):
                outs[i].save(dst_path / f"{i:05d}.png")

        cleanup_memory(verbose=True)

    def eval(self, tasks, **kwargs):
        LOGGER.info("Evaluating compressed model...")

        if kwargs.get("is_check_sparsity", False):
            pass

        results = {}
        tasks = tasks.split(",") if tasks is not None else ""
        if "fid" in tasks:
            scores = self.eval_fid()
            results["fid"] = scores["total"]
            LOGGER.info(f"FID : {results['fid']:.4f}")

        if "lpips" in tasks:
            scores = self.eval_lpips()
            results["lpips"] = scores["total"]
            LOGGER.info(f"LPIPS : {results['lpips']:.4f}")

        if "image_reward" in tasks:
            scores = self.eval_image_reward()
            results["image_reward.BF16"] = scores["origin"]["total"]
            results["image_reward.Quant"] = scores["compress"]["total"]
            LOGGER.info(f"Image Reward(BF16) : {results['image_reward.BF16']:.4f}")
            LOGGER.info(f"Image Reward(Quant) : {results['image_reward.Quant']:.4f}")

        if "clip_score" in tasks:
            scores = self.eval_clip_score()
            results["clip_score.BF16"] = scores["origin"]["total"]
            results["clip_score.Quant"] = scores["compress"]["total"]
            LOGGER.info(f"Clip Score(BF16) : {results['clip_score.BF16']:.4f}")
            LOGGER.info(f"Clip Score(Quant) : {results['clip_score.Quant']:.4f}")

        LOGGER.info("Evaluation complete !")
        return results

    @torch.no_grad()
    def eval_fid(self):
        try:
            from cleanfid import fid

            scores = defaultdict()
            avg_score = 0
            for c in tqdm(self.cate, desc="computing FID..."):
                score = fid.compute_fid(
                    str(self.origin_path / c),
                    str(self.compress_path / c),
                    device=self.device,
                    verbose=False,
                )
                scores[c] = score.item()
                avg_score += score.item()
            scores["total"] = avg_score / len(self.cate)

        except Exception as e:
            LOGGER.error(e)
            raise

        return scores

    @torch.no_grad()
    def eval_lpips(self):
        try:
            import lpips

            loss_fn_alex = lpips.LPIPS(net="vgg")
            scores = defaultdict()
            avg_score = 0
            for c in tqdm(self.cate, desc="computing LPIPS..."):
                org_files = sorted(
                    [str(file) for file in (self.origin_path / c).glob("*.png")]
                )
                cmp_files = sorted(
                    [str(file) for file in (self.compress_path / c).glob("*.png")]
                )
                dist_per_cate = 0
                for i in range(len(org_files)):
                    img0 = lpips.im2tensor(lpips.load_image(org_files[i]))
                    img1 = lpips.im2tensor(lpips.load_image(cmp_files[i]))
                    dist = loss_fn_alex(img0, img1)
                    dist_per_cate += dist.item()
                dist_per_cate /= len(org_files)
                scores[c] = dist_per_cate
                avg_score += dist_per_cate
            scores["total"] = avg_score / len(self.cate)

        except Exception as e:
            LOGGER.error(e)
            raise

        return scores

    @torch.no_grad()
    def eval_image_reward(self):
        try:
            import ImageReward as RM

            model = RM.load("ImageReward-v1.0")

            scores = defaultdict()
            org_scores = defaultdict()
            cmp_scores = defaultdict()
            avg_org_score = avg_cmp_score = 0
            for c in tqdm(self.cate, desc="computing Image Reward..."):
                prompts = self.cate2prmt[c]
                org_files = sorted(
                    [str(file) for file in (self.origin_path / c).glob("*.png")]
                )
                cmp_files = sorted(
                    [str(file) for file in (self.compress_path / c).glob("*.png")]
                )

                org_score_per_cate = cmp_score_per_cate = 0
                for i in range(len(org_files)):
                    k = org_files[i].split(os.sep)[-1].replace(".png", "")
                    p = prompts[k]
                    score = model.score(p, [org_files[i], cmp_files[i]])
                    org_score_per_cate += score[0]
                    cmp_score_per_cate += score[1]

                org_score_per_cate /= len(org_files)
                cmp_score_per_cate /= len(org_files)
                org_scores[c] = org_score_per_cate
                cmp_scores[c] = cmp_score_per_cate
                avg_org_score += org_score_per_cate
                avg_cmp_score += cmp_score_per_cate

            org_scores["total"] = avg_org_score / len(self.cate)
            cmp_scores["total"] = avg_cmp_score / len(self.cate)
            scores["origin"] = org_scores
            scores["compress"] = cmp_scores

        except Exception as e:
            LOGGER.error(e)
            raise

        return scores

    @torch.no_grad()
    def eval_clip_score(self):
        try:
            from torchmetrics.functional.multimodal import clip_score

            clip_score_fn = partial(
                clip_score, model_name_or_path="openai/clip-vit-base-patch16"
            )

            def calculate_clip_score(prompt, img_path1, img_path2):
                image1 = np.array(Image.open(img_path1))[None, ...]
                image2 = np.array(Image.open(img_path2))[None, ...]
                image1 = torch.from_numpy(image1).permute(0, 3, 1, 2)
                image2 = torch.from_numpy(image2).permute(0, 3, 1, 2)
                clip_score1 = clip_score_fn(image1, prompt).detach()
                clip_score2 = clip_score_fn(image2, prompt).detach()
                return [float(clip_score1), float(clip_score2)]

            scores = defaultdict()
            org_scores = defaultdict()
            cmp_scores = defaultdict()
            avg_org_score = avg_cmp_score = 0
            for c in tqdm(self.cate, desc="computing Clip Score..."):
                prompts = self.cate2prmt[c]
                org_files = sorted(
                    [str(file) for file in (self.origin_path / c).glob("*.png")]
                )
                cmp_files = sorted(
                    [str(file) for file in (self.compress_path / c).glob("*.png")]
                )

                org_score_per_cate = cmp_score_per_cate = 0
                for i in range(len(org_files)):
                    k = org_files[i].split(os.sep)[-1].replace(".png", "")
                    p = prompts[k]
                    score = calculate_clip_score(p, org_files[i], cmp_files[i])
                    org_score_per_cate += score[0]
                    cmp_score_per_cate += score[1]
                org_score_per_cate /= len(org_files)
                cmp_score_per_cate /= len(org_files)
                org_scores[c] = org_score_per_cate
                cmp_scores[c] = cmp_score_per_cate
                avg_org_score += org_score_per_cate
                avg_cmp_score += cmp_score_per_cate

            org_scores["total"] = avg_org_score / len(self.cate)
            cmp_scores["total"] = avg_cmp_score / len(self.cate)
            scores["origin"] = org_scores
            scores["compress"] = cmp_scores

        except Exception as e:
            LOGGER.error(e)
            raise

        return scores
