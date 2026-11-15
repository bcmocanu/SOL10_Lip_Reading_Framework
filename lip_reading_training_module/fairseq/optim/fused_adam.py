import types

import torch


def get_fused_adam_class():

    try:
        # The "deprecated" interface in recent versions of apex is a bit
        # faster than the main interface, since we don't use the apex
        # optimizer. This can be installed by passing the
        # `--deprecated_fused_adam` option when building apex.
        global fused_adam_cuda
        import importlib

        fused_adam_cuda = importlib.import_module("fused_adam_cuda")
        return FusedAdamV1
    except ImportError:
        try:
            # fallback to the newer interface
            from apex.optimizers import FusedAdam as _FusedAdam  # noqa
            from apex.multi_tensor_apply import multi_tensor_applier

            if multi_tensor_applier.available:
                return FusedAdamV2
        except ImportError:
            pass
    return None


class FusedAdamV1(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        max_grad_norm=0.0,
        amsgrad=False,
    ):
        global fused_adam_cuda
        import importlib

        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
        }
        super().__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    @property
    def supports_step_with_scale(self):
        return True

    def step(self, closure=None, grads=None, scale=1.0, grad_norms=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        for group, grads_this_group, grad_norm in zip(
            self.param_groups, grads_group, grad_norms
        ):
            if grads_this_group is None:
                grads_this_group = [None] * len(group["params"])

            # compute combined scale factor for this group
            combined_scale = scale
            if group.get("max_grad_norm", 0) > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group["max_grad_norm"]
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group.get("bias_correction", 1) else 0

            for p, grad in zip(group["params"], grads_this_group):
                # note: p.grad should not ever be set for correct
                # operation of mixed precision optimizer that sometimes
                # sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = p.data
                with torch.cuda.device(p.device):
                    fused_adam_cuda.adam(
                        p_data_fp32,
                        out_p,
                        exp_avg,
                        exp_avg_sq,
                        grad,
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        combined_scale,
                        state["step"],
                        self.eps_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss


try:
    from apex.optimizers import FusedAdam
    from apex.multi_tensor_apply import multi_tensor_applier

    class FusedAdamV2(FusedAdam):
        """
        Compared to the original version in Apex, the fairseq version casts grads
        and params to FP32 internally to support ``--memory-efficient-fp16``.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not hasattr(self, "multi_tensor_adam"):
                raise Exception(
                    "Apex installation is outdated. Please install an updated version of apex."
                )

        @property
        def supports_memory_efficient_fp16(self):
            return True

        @property
        def supports_flat_params(self):
            return True

        def step(
            self,
            closure=None,
            grads=None,
            output_params=None,
            scale=None,
            grad_norms=None,
        ):
            """Performs a single optimization step."""
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                bias_correction = 1 if group["bias_correction"] else 0
                beta1, beta2 = group["betas"]

                # assume same step across group now to simplify things
                # per parameter step can be easily support by making it tensor, or pass list into kernel
                if "step" in group:
                    group["step"] += 1
                else:
                    group["step"] = 1

                # create lists for multi-tensor apply
                g_16, p_16, orig_p_16, m_16, v_16 = [], [], [], [], []
                g_32, p_32, m_32, v_32 = [], [], [], []

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "FusedAdam does not support sparse gradients, "
                            "please consider SparseAdam instead"
                        )

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.data, dtype=torch.float
                        )
                    else:
                        state["exp_avg"] = state["exp_avg"].to(
                            device=p.data.device, dtype=torch.float
                        )
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(
                            device=p.data.device, dtype=torch.float
                        )

                    if p.dtype == torch.float16:
                        g_16.append(p.grad.data.float())
                        p_16.append(p.data.float())
                        orig_p_16.append(p.data)
                        m_16.append(state["exp_avg"])
                        v_16.append(state["exp_avg_sq"])
                    elif p.dtype == torch.float32:
                        g_32.append(p.grad.data)
                        p_32.append(p.data)
                        m_32.append(state["exp_avg"])
                        v_32.append(state["exp_avg_sq"])
                    else:
                        raise RuntimeError("FusedAdam only support fp16 and fp32.")

                with torch.cuda.device(p.device):
                    if len(g_16) > 0:
                        multi_tensor_applier(
                            self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_16, p_16, m_16, v_16],
                            group["lr"],
                            beta1,
                            beta2,
                            group["eps"],
                            group["step"],
                            self.adam_w_mode,
                            bias_correction,
                            group["weight_decay"],
                        )
                        for orig_p, p in zip(orig_p_16, p_16):
                            orig_p.copy_(p.data)
                    if len(g_32) > 0:
                        multi_tensor_applier(
                            self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_32, p_32, m_32, v_32],
                            group["lr"],
                            beta1,
                            beta2,
                            group["eps"],
                            group["step"],
                            self.adam_w_mode,
                            bias_correction,
                            group["weight_decay"],
                        )

            return loss


except ImportError:
    pass
