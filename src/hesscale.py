"""HesScale: a diagonal Hessian approximation propagated by a custom backward pass.

Used by UPGD (see optimizers.upgd_optimizer) to obtain the second-order utility
M_l = 1/2 S_l W_l^2 - F_l W_l   (importance of each weight).

Public API
----------
    hesscale(arch_spec, params, x_batch, loss_fn, *aux_batched,
             scalars_batch=None) -> (grads, hess_diag)

Both returned trees match the Haiku `params` dict shape, with each leaf averaged
over the batch dimension.

arch_spec entries
-----------------
- Weight layer:  (path, lin_fn, activation_hint)
    path : either a tuple of dict keys (preferred -- handles Haiku keys that
           contain "/" or "~"), or a slash-separated string (works only when
           no individual key contains "/").
- SCALARS_JOIN  : injects `flatten(h) ++ scalars` between two weight layers;
                  has no parameters of its own. Use when the network
                  concatenates a scalar-feature vector partway through phi
                  (e.g. ForagerNet with hint/last_action/last_reward).

Current assumptions (extend as needed)
--------------------------------------
- Hidden activations are ReLU; output layer is linear.
- `loss_fn(output, *aux)` is per-sample scalar. Any constants in `aux`
  (DQN bootstrap targets, PPO returns/old log-probs) must be stop_gradient'd
  by the caller before being passed in.
"""

import jax
import jax.numpy as jnp


# --- sentinel for the scalars-concat join layer ------------------------------
SCALARS_JOIN = ("__scalars_join__",)


# --- activations -------------------------------------------------------------
def relu(a):     return jnp.maximum(a, 0.0)
def relu_p(a):   return (a > 0.0).astype(a.dtype)        # sigma'
def relu_pp(a):  return jnp.zeros_like(a)                # sigma''  (0 for ReLU)


# --- linear-layer kinds ------------------------------------------------------
def lin_dense(params, h):
    """Dense pre-activation. Accepts 1-D or N-D h (flattens before matmul)."""
    W, b = params
    return h.reshape(-1) @ W + b


def make_conv_lin(stride=1, padding="VALID"):
    """Returns a single-sample Conv2D pre-activation specialized to stride/padding."""
    def lin(params, h):
        W, b = params                                  # h: (H,W,C_in)  W: (kH,kW,C_in,C_out)
        return jax.lax.conv_general_dilated(
            h[None, ...], W,
            window_strides=(stride, stride), padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )[0] + b
    return lin


# --- per-sample forward (with cache) + custom backward -----------------------
def forward_with_cache(layers, x, scalars=None):
    """Run the tagged layer list forward.

    layers : list whose entries are either
        ("weight", lin_fn, (W, b))   -- a parameterized layer
        ("scalars_join",)            -- a flatten + concat-with-scalars join
    scalars : per-sample scalar feature vector, required iff `layers` contains
        a "scalars_join" entry.

    Returns (output, cache). cache entries mirror the layers:
        weight       -> ("weight", h_prev, a)
        scalars_join -> ("scalars_join", h_image_shape, vision_size, scalars_size)
    """
    h, cache = x, []
    last_weight = max(i for i, lay in enumerate(layers) if lay[0] == "weight")
    for l, lay in enumerate(layers):
        if lay[0] == "scalars_join":
            assert scalars is not None, (
                "forward_with_cache: a 'scalars_join' layer is present but "
                "scalars=None was passed."
            )
            h_image = h
            h_flat = h_image.reshape(-1)
            h = jnp.concatenate([h_flat, scalars], axis=-1)
            cache.append(("scalars_join", h_image.shape, h_flat.shape[0], scalars.shape[0]))
        else:                                       # ("weight", lin_fn, params)
            _, lin, params = lay
            a = lin(params, h)
            cache.append(("weight", h, a))
            if l == last_weight:
                out = a                              # linear output
            else:
                h = relu(a)
    return out, cache


def output_signals(loss_fn, output, *aux):
    """Exact first- and second-order seeds at the network output.
    s_L = diag(jax.hessian(loss_fn)) -- exact, regardless of loss type."""
    g_L = jax.grad(loss_fn)(output, *aux)
    H_L = jax.hessian(loss_fn)(output, *aux)
    return g_L, jnp.diagonal(H_L)


def backward_hesscale(layers, cache, G_L, S_L):
    """HesScale backward recursion.

    Carry semantics: (G, S) are kept "wrt the post-activation h" of the
    previous layer's output. At each weight layer we (a) convert wrt h to
    wrt a using sigma', sigma'' of the layer's OWN pre-activation, (b)
    compute weight grads/hess from g, s, (c) propagate dh, Wt_s back as
    the new G, S. At a scalars_join we just slice off the vision portion
    of G/S and reshape it back to the image shape -- no parameter update.
    """
    L = len(layers)
    grads, hess = [None] * L, [None] * L
    G, S = G_L, S_L
    last_weight = max(i for i, lay in enumerate(layers) if lay[0] == "weight")

    for l in reversed(range(L)):
        lay = layers[l]

        # --- join: split off vision portion of the carry, reshape ---
        if lay[0] == "scalars_join":
            _, h_image_shape, vision_size, _ = cache[l]
            G = G[:vision_size].reshape(h_image_shape)
            S = S[:vision_size].reshape(h_image_shape)
            continue

        # --- weight layer ---
        _, lin, (W, b) = lay
        _, h_prev, a = cache[l]

        # (a) Convert carry wrt h_l to wrt a_l. Output layer is linear, so they coincide.
        if l == last_weight:
            g, s = G, S
        else:
            sp  = relu_p(a)
            spp = relu_pp(a)
            g = sp * G                                # = g_l in HesScale notation
            s = sp ** 2 * S + spp * G                 # = s_l (Eq. 4 with G playing dh's role)

        # (b) Eq. 1 and Eq. 3 -- weight grads / hess via two VJPs
        _, vjp_fn   = jax.vjp(lin, (W, b), h_prev)
        d_params, _ = vjp_fn(g)
        grads[l]    = d_params

        _, vjp_h_sq = jax.vjp(lin, (W, b), h_prev ** 2)
        S_params, _ = vjp_h_sq(s)
        hess[l]     = S_params

        # (c) Propagate to h_prev (input of this layer = output of previous).
        _, vjp_fn_g    = jax.vjp(lin, (W, b), h_prev)
        _, dh          = vjp_fn_g(g)
        _, vjp_W_sq    = jax.vjp(lin, (W ** 2, b), h_prev)
        _, Wt_s        = vjp_W_sq(s)
        G, S = dh, Wt_s

    return grads, hess


# --- Haiku params <-> layer-list adapter -------------------------------------
def _path_get(d, path):
    """Walk a nested dict. `path` can be:
       - tuple of keys (preferred -- handles keys containing '/' or '~')
       - slash-separated string (only safe when no key contains '/')."""
    if isinstance(path, tuple):
        for k in path:
            d = d[k]
        return d
    if isinstance(path, str):
        if path == "":
            return d
        for k in path.split("/"):
            d = d[k]
        return d
    raise TypeError(f"path must be tuple or str, got {type(path).__name__}")


def _path_set(d, path, value):
    """Set value at a path in a nested dict (creates intermediate dicts)."""
    if isinstance(path, tuple):
        keys = list(path)
    elif isinstance(path, str):
        keys = path.split("/")
    else:
        raise TypeError(f"path must be tuple or str, got {type(path).__name__}")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def haiku_to_layers(params, arch_spec):
    """arch_spec : list of (path, lin_fn, activation_hint) and/or SCALARS_JOIN.
    Returns the tagged layer list that forward/backward consume."""
    out = []
    for entry in arch_spec:
        if entry == SCALARS_JOIN:
            out.append(("scalars_join",))
        else:
            path, lin_fn, _ = entry
            sub = _path_get(params, path)
            out.append(("weight", lin_fn, (sub["w"], sub["b"])))
    return out


def layers_grads_to_haiku_dict(layer_grads, arch_spec):
    """Inverse of haiku_to_layers: rebuild the same nested-dict shape as the input params.
    Join entries have no params and are skipped."""
    out = {}
    for entry, grad in zip(arch_spec, layer_grads):
        if entry == SCALARS_JOIN:
            continue
        path, _, _ = entry
        dW, db = grad
        _path_set(out, path, {"w": dW, "b": db})
    return out


# --- public entrypoint -------------------------------------------------------
def hesscale(arch_spec, params, x_batch, loss_fn, *aux_batched, scalars_batch=None):
    """Batched HesScale.

    Args:
        arch_spec : list of (path, lin_fn, activation_hint) and/or SCALARS_JOIN.
        params    : Haiku-style nested params dict (state.params).
        x_batch   : (B, ...) inputs.
        loss_fn   : per-sample scalar loss_fn(output, *aux).
        aux_batched : each (B, ...) auxiliary (target, action, ...).
                      Must be stop_gradient'd by the caller.
        scalars_batch : (B, scalars_size) -- required iff arch_spec contains SCALARS_JOIN.

    Returns:
        (grads, hess_diag): two Haiku-shaped dicts matching `params`, averaged over batch.
    """
    layers = haiku_to_layers(params, arch_spec)
    has_join = any(lay[0] == "scalars_join" for lay in layers)
    if has_join and scalars_batch is None:
        raise ValueError(
            "arch_spec contains SCALARS_JOIN but scalars_batch=None was passed."
        )

    if has_join:
        def per_sample_with_scalars(x, scalars, *aux):
            out, cache = forward_with_cache(layers, x, scalars=scalars)
            g_L, s_L   = output_signals(loss_fn, out, *aux)
            return backward_hesscale(layers, cache, g_L, s_L)
        in_axes = (0, 0) + (0,) * len(aux_batched)
        grads_b, hess_b = jax.vmap(per_sample_with_scalars, in_axes=in_axes)(
            x_batch, scalars_batch, *aux_batched
        )
    else:
        def per_sample_no_scalars(x, *aux):
            out, cache = forward_with_cache(layers, x)
            g_L, s_L   = output_signals(loss_fn, out, *aux)
            return backward_hesscale(layers, cache, g_L, s_L)
        in_axes = (0,) + (0,) * len(aux_batched)
        grads_b, hess_b = jax.vmap(per_sample_no_scalars, in_axes=in_axes)(
            x_batch, *aux_batched
        )

    grads = jax.tree.map(lambda t: t.mean(0), grads_b)
    hess  = jax.tree.map(lambda t: t.mean(0), hess_b)
    return (
        layers_grads_to_haiku_dict(grads, arch_spec),
        layers_grads_to_haiku_dict(hess,  arch_spec),
    )
