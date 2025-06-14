import sys
from pathlib import Path

import torch
from tqdm import tqdm
from lm_eval import models, evaluator
from lm_eval.tasks import TaskManager, get_task_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import LOGGER  # noqa: E402
from utils.dataset import get_loaders  # noqa: E402
from utils.module import check_sparsity  # noqa: E402


class LMEvaluator:
    def __init__(self, device, n_samples=None):
        self.device = device
        self.n_samples = n_samples

    def eval(self, model, tasks, **kwargs):
        LOGGER.info("Evaluating compressed model...")
        check_sparsity(model, self.device)

        model.to(self.device)
        results = {}
        tasks = tasks.split(",")
        if "ppl" in tasks:
            datasets = ["wikitext2", "ptb", "c4"]
            tokenizer_path = kwargs.get("tokenizer_path")
            seq_len = kwargs.get("seq_len", 2048)
            ppl = self.eval_ppl(
                model=model,
                tokenizer_path=tokenizer_path,
                datasets=datasets,
                seq_len=seq_len,
            )
            results.update(ppl)
            tasks.remove("ppl")

        batch_size = kwargs.get("batch_size", 1)
        acc = self.eval_QA(model=model, tasks=tasks, batch_size=batch_size)
        results.update(acc)
        LOGGER.info("Evaluation complete !")
        return results

    def eval_ppl(self, model, tokenizer_path, datasets, seq_len):
        model.eval()
        ppl = {}
        for dataset in datasets:
            # try:
            _, testenc = get_loaders(name=dataset, tokenizer_path=tokenizer_path)
            ppl[f"ppl.{dataset}"] = self.compute_ppl(
                model=model, dataset=testenc, seq_len=seq_len
            )

            LOGGER.info(f"PPL[{dataset.upper()}] : {ppl[f'ppl.{dataset}']:.4f}")
            # except Exception as e:
            #     ppl[f"ppl.{dataset}"] = sys.maxsize
            #     LOGGER.error(e)
        return ppl

    @torch.no_grad()
    def compute_ppl(self, model, dataset, seq_len=2048):
        input_ids = dataset.input_ids.to(self.device)
        n_samples = input_ids.numel() // seq_len
        if self.n_samples is not None:
            n_samples = min(n_samples, self.n_samples)

        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss()
        for i in tqdm(range(n_samples), desc="Evaluating..."):
            batch = input_ids[:, (i * seq_len) : ((i + 1) * seq_len)]
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, (i * seq_len) : ((i + 1) * seq_len)][:, 1:]
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seq_len
            nlls.append(neg_log_likelihood)

        return (torch.exp(torch.stack(nlls).sum() / (n_samples * seq_len))).item()

    def eval_QA(self, model, tasks, batch_size):
        results = {}
        for task in tasks:
            try:
                acc = self.compute_zeroshot(
                    model=model, task=task, batch_size=batch_size
                )
                if task == "lambada":
                    results[task] = acc["results"]["lambada_openai"]["acc,none"] * 100
                elif task == "truthfulqa":
                    results[task] = acc["results"]["truthfulqa_mc1"]["acc,none"] * 100
                else:
                    results[task] = acc["results"][f"{task}"]["acc,none"] * 100

                LOGGER.info(f"QA[{task.upper()}] : {results[task]:.4f}")

            except Exception as e:
                results[task] = sys.maxsize
                LOGGER.error(e)

        return results

    @torch.no_grad()
    def compute_zeroshot(
        self, model, task, batch_size, num_fewshot=0, fewshot_random_seed=1234
    ):
        task_manager = TaskManager()
        task_dict = get_task_dict(
            task,
            task_manager,
        )

        # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
        # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
        def _adjust_config(task_dict):
            adjusted_task_dict = {}
            for task_name, task_obj in task_dict.items():
                if isinstance(task_obj, dict):
                    adjusted_task_dict = {
                        **adjusted_task_dict,
                        **{task_name: _adjust_config(task_obj)},
                    }
                else:
                    # override tasks' fewshot values to the provided num_fewshot arg value
                    # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                    if num_fewshot is not None:
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)

                    # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                    task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                    adjusted_task_dict[task_name] = task_obj

            return adjusted_task_dict

        task_dict = _adjust_config(task_dict)
        results = evaluator.evaluate(
            lm=models.huggingface.HFLM(pretrained=model, batch_size=batch_size),
            task_dict=task_dict,
            limit=self.n_samples,
            cache_requests=True,
            log_samples=False,
            verbosity="ERROR",
        )
        return results


if __name__ == "__main__":
    from models.opt import CompressOPTForCausalLM

    model_path = r"d:\\models\\opt-350M"
    model = CompressOPTForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lm_evaluator = LMEvaluator(device=model.device, n_samples=50)
    kwargs = {"tokenizer_path": model_path, "seq_len": 2048, "batch_size": 8}
    results = lm_evaluator.eval(model, tasks="ppl,boolq,arc_easy", **kwargs)
    print(results)
