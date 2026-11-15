from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import logging
import math
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

logger = logging.getLogger("training")


def validate_schedule_config(cfg) -> None:
    if int(cfg.optimization.max_epoch) < 1:
        raise ValueError("optimization.max_epoch trebuie sa fie pozitiv.")
    if int(cfg.optimization.max_update) < 0:
        raise ValueError("optimization.max_update trebuie sa fie 0 (automat) sau pozitiv.")
    rates = list(cfg.optimization.lr)
    if len(rates) != 1 or not math.isfinite(rates[0]) or rates[0] <= 0:
        raise ValueError("Configurati o singura rata de invatare pozitiva in optimization.lr.")
    schedule = cfg.lr_scheduler
    if schedule._name == "tri_stage":
        ratios = schedule.phase_ratio
        if (ratios is None or len(ratios) != 3
                or any(not math.isfinite(r) or r < 0 for r in ratios)
                or ratios[2] <= 0 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9)):
            raise ValueError("phase_ratio necesita trei proportii nenegative, cu suma 1 si decay pozitiv.")
        for key in ("init_lr_scale", "final_lr_scale"):
            if not 0 < float(schedule[key]) <= 1:
                raise ValueError(f"lr_scheduler.{key} trebuie sa fie in intervalul (0, 1].")
    elif schedule._name == "cosine":
        if int(schedule.warmup_updates) < 0 or not 0 <= schedule.min_lr < rates[0]:
            raise ValueError("Cosine necesita warmup_updates >= 0 si 0 <= min_lr < lr.")
        if float(schedule.t_mult) != 1.0 or float(schedule.lr_period_updates) > 0:
            raise ValueError("Perioada cosine este calculata automat din bugetul antrenarii.")
    else:
        raise ValueError("Profilul necesita schedulerul tri_stage (stage 1) sau cosine (stage 2).")


@contextmanager
def _preserve_random_state():
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _manifest_identity(cfg) -> dict[str, str]:
    split = cfg.dataset.train_subset
    paths = {"tsv": Path(cfg.task.data) / f"{split}.tsv"}
    label_dir = Path(cfg.task.label_dir or cfg.task.data)
    labels = set(cfg.task.labels)
    if cfg.model._name == "zero-avsr":
        labels.add("ltr")
    for label in sorted(labels):
        paths[label] = label_dir / f"{split}.{label}"
    result = {}
    for suffix, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[suffix] = digest.hexdigest()
    return result


def _request(cfg) -> dict:
    scheduler = OmegaConf.to_container(cfg.lr_scheduler, resolve=True, enum_to_str=True)
    scheduler.pop("max_update", None)
    scheduler.pop("lr", None)
    return {
        "model": cfg.model._name,
        "max_epoch": int(cfg.optimization.max_epoch),
        "max_update_cap": int(cfg.optimization.max_update),
        "update_freq": [int(n) for n in cfg.optimization.update_freq],
        "lr": list(cfg.optimization.lr),
        "scheduler": scheduler,
        "batching": {
            "seed": int(cfg.common.seed),
            "max_tokens": cfg.dataset.max_tokens,
            "batch_size": cfg.dataset.batch_size,
            "required_batch_size_multiple": cfg.dataset.required_batch_size_multiple,
            "max_sample_size": cfg.task.max_sample_size,
            "min_sample_size": cfg.task.min_sample_size,
            "max_trim_sample_size": cfg.task.max_trim_sample_size,
            "pad_audio": cfg.task.pad_audio,
            "modalities": list(cfg.task.modalities),
        },
        "train_manifests": _manifest_identity(cfg),
    }


def _count_batches(trainer) -> tuple[int, int]:
    """Inspect the sampler only: no __getitem__, collater or DataLoader iteration."""
    from fairseq import utils

    cfg, task = trainer.cfg, trainer.task
    split = cfg.dataset.train_subset
    if trainer.data_parallel_world_size != 1 or task.has_sharded_data(split):
        raise ValueError("Planificarea este disponibila pentru un singur GPU si un dataset local.")
    previous_datasets, previous_iterators = task.datasets, task.dataset_to_epoch_iter
    with _preserve_random_state():
        try:
            task.datasets, task.dataset_to_epoch_iter = dict(previous_datasets), {}
            task.load_dataset(split, epoch=1, combine=True)
            dataset = task.dataset(split)
            if not task.can_reuse_epoch_itr(dataset):
                raise ValueError("Planificarea necesita un dataset cu lungimi stabile intre epoci.")
            epoch_itr = task.get_batch_iterator(
                dataset=dataset,
                max_tokens=cfg.dataset.max_tokens,
                max_sentences=cfg.dataset.batch_size,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(), trainer.model.max_positions(), cfg.dataset.max_tokens),
                ignore_invalid_inputs=True,
                required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
                seed=cfg.common.seed,
                num_shards=1, shard_id=0, num_workers=0, epoch=1,
                data_buffer_size=0, disable_iterator_cache=True,
            )
            batches = len(epoch_itr)
            examples = sum(len(batch) for batch in epoch_itr.frozen_batches)
            if batches < 1:
                raise ValueError("Nu exista minibatchuri dupa filtrarea datelor de antrenare.")
            return batches, examples
        finally:
            task.datasets, task.dataset_to_epoch_iter = previous_datasets, previous_iterators


def _new_plan(request: dict, batches: int, examples: int) -> dict:
    freq = request["update_freq"]
    updates = [math.ceil(batches / freq[min(epoch, len(freq) - 1)])
               for epoch in range(request["max_epoch"])]
    epoch_budget = sum(updates)
    cap = request["max_update_cap"]
    total = min(epoch_budget, cap) if cap > 0 else epoch_budget
    scheduler = copy.deepcopy(request["scheduler"])
    scheduler.update(max_update=total, lr=list(request["lr"]))
    if scheduler["_name"] == "tri_stage":
        ratios = scheduler["phase_ratio"]
        warmup, hold = int(total * ratios[0]), int(total * ratios[1])
        # Assign rounding remainder to decay so the endpoint is the planned budget.
        scheduler.update(phase_ratio=None, warmup_steps=warmup, hold_steps=hold,
                         decay_steps=total - warmup - hold)
        if scheduler["decay_steps"] < 1:
            raise ValueError("Buget prea mic pentru fazele tri_stage configurate.")
    else:
        warmup = int(scheduler["warmup_updates"])
        if total <= warmup:
            raise ValueError(f"Bugetul de {total} actualizari nu depaseste warmup_updates={warmup}. "
                             "Mariti max_epoch sau reduceti warmup_updates in stage2.yaml.")
        scheduler["lr_period_updates"] = total - warmup
        if scheduler["warmup_init_lr"] < 0:
            scheduler["warmup_init_lr"] = scheduler["min_lr"]
    return {
        "version": 1,
        "request": request,
        "batches_per_epoch": batches,
        "examples_per_epoch": examples,
        "updates_per_epoch": updates,
        "epoch_update_budget": epoch_budget,
        "total_updates": total,
        "scheduler": scheduler,
    }


def prepare_training_plan(trainer, saved_plan: dict | None = None) -> dict:
    """Resolve a new plan or restore its exact configuration before optimizer creation."""
    cfg = trainer.cfg
    validate_schedule_config(cfg)
    logger.info("Verificare manifeste pentru planul de antrenare...")
    request = _request(cfg)
    if saved_plan is None:
        logger.info("Planificare minibatchuri din metadate, fara decodare video/audio...")
        batches, examples = _count_batches(trainer)
        plan = _new_plan(request, batches, examples)
    else:
        if not isinstance(saved_plan, dict) or saved_plan.get("version") != 1:
            raise ValueError("Plan de antrenare lipsa sau incompatibil in checkpoint.")
        if saved_plan.get("request") != request:
            raise ValueError("Configuratia schedulerului, batchingul sau manifestele TRAIN difera de "
                             "checkpoint. La reluare pastrati configuratia si datele initiale; "
                             "pentru un experiment nou folositi un director de rezultate gol.")
        plan = copy.deepcopy(saved_plan)
        expected = _new_plan(request, plan["batches_per_epoch"], plan["examples_per_epoch"])
        if plan != expected:
            raise ValueError("Planul salvat nu este coerent cu configuratia din checkpoint.")
    with open_dict(cfg), open_dict(cfg.lr_scheduler):
        cfg.optimization.max_update = plan["total_updates"]
        for key, value in plan["scheduler"].items():
            cfg.lr_scheduler[key] = value
    _log_plan(plan, resumed=saved_plan is not None)
    return plan


def _log_plan(plan: dict, resumed: bool) -> None:
    request, schedule = plan["request"], plan["scheduler"]
    updates = plan["updates_per_epoch"]
    logger.info("Plan %s: %d exemple, %d minibatchuri/epoca, acumulare=%s; "
                "%d..%d actualizari planificate/epoca, max_epoch=%d, buget=%d actualizari.",
                "restaurat" if resumed else "nou", plan["examples_per_epoch"],
                plan["batches_per_epoch"], request["update_freq"], min(updates), max(updates),
                request["max_epoch"], plan["total_updates"])
    if request["max_update_cap"] > 0:
        logger.info("Limita suplimentara max_update=%d; buget calculat din epoci=%d.",
                    request["max_update_cap"], plan["epoch_update_budget"])
    if schedule["_name"] == "tri_stage":
        warmup, hold = schedule["warmup_steps"], schedule["hold_steps"]
        peak = request["lr"][0]
        logger.info("LR tri_stage: warmup 0..%d, mentinere %d..%d, decay %d..%d; "
                    "initial=%.8g, maxim=%.8g, final=%.8g.",
                    warmup, warmup, warmup + hold, warmup + hold, plan["total_updates"],
                    peak * schedule["init_lr_scale"], peak, peak * schedule["final_lr_scale"])
    else:
        logger.info("LR cosine: warmup=%d, perioada=%d actualizari; "
                    "initial=%.8g, maxim=%.8g, min_lr=%.8g.",
                    schedule["warmup_updates"], schedule["lr_period_updates"],
                    schedule["warmup_init_lr"], request["lr"][0], schedule["min_lr"])
    logger.info("Schedulerul avanseaza la actualizari reusite; overflow/OOM poate reduce "
                "numarul executat fata de plan. Planul este pastrat in checkpoint.")
