"""Analytical FP32 tensor working-set formula used in the paper."""

def _estimated_working_memory_bytes(
    dataset,
    optimizer,
    batch_size,
    hidden1=256,
    hidden2=64,
    num_classes=10,
):
    """
    Approximate peak live FP32 tensor working set, excluding resident full data.

    SGD:
      minibatch + parameters + gradients + saved tanh activations + output grad.

    Adam:
      SGD estimate + first/second-moment parameter states.

    ER-1:
      minibatch + parameters + explicit (S,Z) forward cache + propagated deltas
      + centered first-layer local-regression tensors + local update
      + largest auto-ridge square system and Cholesky factor.

    This intentionally omits allocator/library workspaces, so it is an
    interpretable shape-based estimate rather than claimed CUDA peak memory.
    """
    input_dim = {
        "mnist": 784,
        "fashionmnist": 784,
        "cifar10": 3072,
    }[str(dataset)]

    B = int(batch_size)
    h1 = int(hidden1)
    h2 = int(hidden2)
    C = int(num_classes)

    f32 = 4
    i64 = 8

    # Parameters, including bias.
    P = (
        input_dim * h1 + h1
        + h1 * h2 + h2
        + h2 * C + C
    )
    params = P * f32

    # index_select/current-batch working tensors: input, bipolar targets, labels.
    minibatch = (
        B * input_dim * f32
        + B * C * f32
        + B * i64
    )

    # Tanh outputs saved for autograd-based baselines.
    saved_activations = B * (h1 + h2 + C) * f32
    output_grad = B * C * f32

    opt = str(optimizer).lower()

    if opt == "sgd":
        # params + gradients
        return (
            minibatch
            + params
            + params
            + saved_activations
            + output_grad
        )

    if opt == "adam":
        # SGD-like live tensors + first and second moment states.
        return (
            minibatch
            + params
            + params
            + saved_activations
            + output_grad
            + 2 * params
        )

    if opt == "er1":
        # Explicit pre-/post-activation cache.
        forward_cache = 2 * B * (h1 + h2 + C) * f32
        propagated_deltas = B * (h1 + h2 + C) * f32

        # The first-layer local regression dominates at B=4096.
        in_dim = input_dim
        out_dim = h1
        centered = B * (in_dim + out_dim) * f32
        dW = in_dim * out_dim * f32

        # Auto ridge chooses the smaller primal/dual square system.
        system_dim = min(B, in_dim)
        system_and_cholesky = (
            2 * system_dim * system_dim * f32
        )

        if in_dim <= B:  # primal
            rhs = in_dim * out_dim * f32
        else:            # dual
            rhs = B * out_dim * f32

        return (
            minibatch
            + params
            + forward_cache
            + propagated_deltas
            + centered
            + dW
            + system_and_cholesky
            + rhs
        )

    raise ValueError(opt)
