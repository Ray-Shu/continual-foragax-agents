import json
import jax
import jax.numpy as jnp
from algorithms.PPORegistry import getAgent as getPPOAgent
from algorithms.registry import getAgent
from experiment.ExperimentModel import load as load_exp
from foragax.registry import make as make_env
from ml_instrumentation.Collector import Collector

FOV = 9
OBS_SHAPE = (FOV, FOV, 3)
ACTIONS = 4
BASE = f"experiments/E136-big/foragax-sweep/ForagaxBig-v4/{FOV}"

CLASSDEFS = """classDef styleInput fill:#fbcfe8,stroke:#f472b6,color:#0f172a,font-family:Inter,sans-serif
classDef styleNorm fill:#fef9c3,stroke:#facc15,color:#0f172a,font-family:Inter,sans-serif
classDef styleMlp fill:#e0f2fe,stroke:#38bdf8,color:#0f172a,font-family:Inter,sans-serif
classDef styleRnn fill:#ffedd5,stroke:#fb923c,color:#0f172a,font-family:Inter,sans-serif
classDef styleHead fill:#fae8ff,stroke:#e879f9,color:#0f172a,font-family:Inter,sans-serif
classDef styleConcat fill:#f8fafc,stroke:#64748b,stroke-dasharray: 4 4,color:#0f172a,font-family:Inter,sans-serif"""


def p(n):
    """Format param count."""
    return f"{n:,}"


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def params_by_module(params_dict):
    """Flatten nested Flax params dict to {module_name: count}."""
    result = {}

    def _walk(d, prefix=()):
        for k, v in d.items():
            path = prefix + (k,)
            if isinstance(v, dict):
                _walk(v, path)
            else:
                mod = "/".join(prefix) if prefix else k
                result[mod] = result.get(mod, 0) + int(v.size)

    top = params_dict.get("params", params_dict)
    _walk(top)
    return result


def graph_to_mermaid(graph):
    """Convert a graph dict to a Mermaid flowchart string."""
    in_sg = {}

    def map_sg(sgs):
        for sg in sgs:
            for nid in sg.get("nodes", []):
                in_sg[nid] = sg["id"]
            if "subgraphs" in sg:
                map_sg(sg["subgraphs"])

    map_sg(graph.get("subgraphs", []))

    lines = ["graph TD", "classDef default font-family:Inter,sans-serif,font-size:12px"]
    lines.append(CLASSDEFS)

    # Top-level nodes (not in any subgraph)
    for nid, node in graph["nodes"].items():
        if nid not in in_sg:
            style = node.get("type", "styleMlp")
            lines.append(f'    {nid}("{node["label"]}"):::{style}')

    # Subgraphs (recursive)
    def render_sg(sg, indent=4):
        sp = " " * indent
        lines.append(f'{sp}subgraph {sg["id"]} ["{sg["label"]}"]')
        lines.append(f"{sp}direction TB")
        for nid in sg.get("nodes", []):
            node = graph["nodes"][nid]
            style = node.get("type", "styleMlp")
            lines.append(f'{sp}{nid}("{node["label"]}"):::{style}')

        for sub in sg.get("subgraphs", []):
            render_sg(sub, indent + 4)

        # Intra-subgraph edges
        for a, b in graph["edges"]:
            if in_sg.get(a) == sg["id"] and in_sg.get(b) == sg["id"]:
                lines.append(f"{sp}{a} --> {b}")
        lines.append(f"{sp}end")

    for sg in graph.get("subgraphs", []):
        render_sg(sg)

    # Cross-subgraph and external edges
    for a, b in graph["edges"]:
        if in_sg.get(a) != in_sg.get(b) or not in_sg.get(a):
            lines.append(f"    {a} --> {b}")

    # Invisible ordering edges
    for a, b in graph.get("invisible_edges", []):
        lines.append(f"    {a} ~~~ {b}")

    return "\n".join(lines)


def _lbl(name, dims=None, params=None):
    """Build a node label: Name / dims / params.
    Avoids parentheses in output since they confuse the Mermaid parser.
    """
    parts = [name]
    if dims:
        # Replace () with [] to avoid Mermaid parse errors
        parts.append(dims.replace("(", "[").replace(")", "]"))
    if params:
        parts.append(f"{p(params)} params")
    return "\\n".join(parts)


# ── DQN graph builders ────────────────────────────────────────────────────────


def build_dqn_graph(variant_data):
    m = variant_data["modules"]
    use_ln = variant_data["use_layernorm"]
    has_rt = variant_data["has_rt"]
    has_hint = variant_data.get("has_hint", False)
    conv_out = variant_data["conv_out"]
    concat_sz = variant_data["concat_size"]
    scalars = variant_data["scalars"]
    fov = variant_data["fov"]

    # scalars: Actions[4] + Reward[1] + RT[1 if has_rt] + Hint[4 if has_hint]
    a_bits = 4
    r_bits = 1
    rt_bits = 1 if has_rt else 0
    h_bits = 4 if has_hint else 0

    scalar_label = f"A[{a_bits}], R[{r_bits}]"
    if has_rt:
        scalar_label += f", RT[{rt_bits}]"
    if has_hint:
        scalar_label += f", Hint[{h_bits}]"

    scalar_dims = f"[{scalars}]"

    conv_p = m.get("phi/phi/~/phi", 0)
    d1_p = m.get("phi/phi/~/linear", 0)
    d2_p = m.get("phi/phi/~/linear_1", 0)
    qh_p = m.get("q/q", 0)

    nodes = {
        "Obs": {
            "label": _lbl("Observation", f"{fov}×{fov}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, scalar_dims),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Conv": {
            "label": _lbl("Conv2D 16×3×3", f"{fov}×{fov}×3→{fov}×{fov}×16", conv_p),
            "type": "styleMlp",
        },
        "ReLU1": {"label": "ReLU", "type": "styleNorm"},
        "Flat": {"label": _lbl("Flatten", f"→{conv_out}"), "type": "styleMlp"},
        "Concat": {"label": _lbl("Concat", f"→{concat_sz}"), "type": "styleConcat"},
        "Dense1": {"label": _lbl("Dense", f"{concat_sz}→64", d1_p), "type": "styleMlp"},
        "Dense2": {"label": _lbl("Dense", "64→64", d2_p), "type": "styleMlp"},
        "QHead": {"label": _lbl("Q-Head", "64→4", qh_p), "type": "styleHead"},
    }
    edges = [
        ("Obs", "Conv"),
        ("Conv", "ReLU1"),
        ("ReLU1", "Flat"),
        ("Flat", "Concat"),
        ("Scalars", "Concat"),
        ("Concat", "Dense1"),
    ]
    repr_nodes = ["Dense1"]

    if use_ln:
        ln1_p = m.get("phi/phi/~/layer_norm", 0)
        ln2_p = m.get("phi/phi/~/layer_norm_1", 0)
        nodes["LN1"] = {"label": _lbl("LayerNorm", "64", ln1_p), "type": "styleNorm"}
        nodes["Act1"] = {"label": "ReLU", "type": "styleNorm"}
        nodes["LN2"] = {"label": _lbl("LayerNorm", "64", ln2_p), "type": "styleNorm"}
        nodes["Act2"] = {"label": "ReLU", "type": "styleNorm"}
        edges += [
            ("Dense1", "LN1"),
            ("LN1", "Act1"),
            ("Act1", "Dense2"),
            ("Dense2", "LN2"),
            ("LN2", "Act2"),
            ("Act2", "QHead"),
        ]
        repr_nodes += ["LN1", "Act1", "Dense2", "LN2", "Act2"]
    else:
        nodes["Act1"] = {"label": "ReLU", "type": "styleNorm"}
        nodes["Act2"] = {"label": "ReLU", "type": "styleNorm"}
        edges += [
            ("Dense1", "Act1"),
            ("Act1", "Dense2"),
            ("Dense2", "Act2"),
            ("Act2", "QHead"),
        ]
        repr_nodes += ["Act1", "Dense2", "Act2"]

    # Reorder nodes so subgraph content appears contiguous
    ordered_nodes = {}
    top_keys = ["Obs", "Scalars", "Conv", "ReLU1", "Flat", "Concat"]
    for k in top_keys:
        ordered_nodes[k] = nodes[k]
    for k in repr_nodes:
        ordered_nodes[k] = nodes[k]
    ordered_nodes["QHead"] = nodes["QHead"]

    return {
        "nodes": ordered_nodes,
        "edges": edges,
        "subgraphs": [
            {
                "id": "ForagerNet",
                "label": "ForagerNet (Representation)",
                "nodes": repr_nodes,
            }
        ],
    }


# ── DRQN graph builders ───────────────────────────────────────────────────────


def build_drqn_graph(variant_data):
    m = variant_data["modules"]
    use_ln = variant_data["use_layernorm"]
    conv_out = variant_data["conv_out"]
    concat_sz = variant_data["concat_size"]
    pre = variant_data.get("pre_core_layers", variant_data.get("pre_gru_layers", 0))
    post = variant_data.get("post_core_layers", variant_data.get("post_gru_layers", 0))
    fov = variant_data["fov"]
    has_hint = variant_data.get("has_hint", False)

    conv_p = m.get("phi/ForagerGRUNetReLU/~/phi", 0)
    gru_p = m.get("phi/ForagerGRUNetReLU/~/gru", 0) + m.get(
        "phi/ForagerGRUNetReLU/~/gru/~/gru_inner", 0
    )
    qh_p = m.get("q/q", 0)

    # GRU input and output dims
    if pre > 0:
        pre_p = m.get("phi/ForagerGRUNetReLU/~/linear", 0)
        gru_in = 64
        skip_sz = 64 + 64  # 128 (gru_out + pre_out)
    else:
        pre_p = 0
        gru_in = concat_sz
        skip_sz = 64 + concat_sz

    scalar_label = "A[4], R[1]"
    if has_hint:
        scalar_label += ", Hint[4]"

    nodes = {
        "Obs": {
            "label": _lbl("Seq Obs", f"{fov}×{fov}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, f"[{variant_data['scalars']}]"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Conv": {
            "label": _lbl(
                "Conv2D 16×3×3",
                f"{fov}×{fov}×3→{fov}×{fov}×16",
                conv_p,
            ),
            "type": "styleMlp",
        },
        "ReLU1": {"label": "ReLU", "type": "styleNorm"},
        "Flat": {"label": _lbl("Flatten", f"→{conv_out}"), "type": "styleMlp"},
        "Concat": {"label": _lbl("Concat", f"→{concat_sz}"), "type": "styleConcat"},
        "GRU": {"label": _lbl("GRU 64", f"{gru_in}→64", gru_p), "type": "styleRnn"},
        "ReLU2": {"label": "ReLU", "type": "styleNorm"},
        "Skip": {"label": _lbl("Concat Skip", f"→{skip_sz}"), "type": "styleConcat"},
    }
    edges = [
        ("Obs", "Conv"),
        ("Conv", "ReLU1"),
        ("ReLU1", "Flat"),
        ("Flat", "Concat"),
        ("Scalars", "Concat"),
    ]

    pre_nodes = []
    if pre > 0:
        ln_p = m.get("phi/ForagerGRUNetReLU/~/layer_norm", 0) if use_ln else 0
        nodes["Pre"] = {
            "label": _lbl("Dense", f"{concat_sz}→64", pre_p),
            "type": "styleMlp",
        }
        pre_nodes.append("Pre")
        if use_ln:
            nodes["PreLN"] = {
                "label": _lbl("LayerNorm", "64", ln_p),
                "type": "styleNorm",
            }
            nodes["PreAct"] = {"label": "ReLU", "type": "styleNorm"}
            edges += [
                ("Concat", "Pre"),
                ("Pre", "PreLN"),
                ("PreLN", "PreAct"),
                ("PreAct", "GRU"),
                ("PreAct", "Skip"),
            ]
            pre_nodes += ["PreLN", "PreAct"]
        else:
            nodes["PreAct"] = {"label": "ReLU", "type": "styleNorm"}
            edges += [
                ("Concat", "Pre"),
                ("Pre", "PreAct"),
                ("PreAct", "GRU"),
                ("PreAct", "Skip"),
            ]
            pre_nodes.append("PreAct")
    else:
        edges += [("Concat", "GRU"), ("Concat", "Skip")]

    edges += [("GRU", "ReLU2"), ("ReLU2", "Skip")]
    curr_out = "Skip"

    post_nodes = []
    if post > 0:
        curr_in_sz = skip_sz
        # ForagerGRUNetReLU post_gru_mlp
        for i in range(post):
            suffix = f"_{i}" if i > 0 else ""
            linear_key = f"phi/ForagerGRUNetReLU/~/linear{suffix}"
            post_p = m.get(linear_key, 0)
            if post_p == 0:
                linear_key = f"phi/ForagerGRUNetReLU/~/post_gru_mlp/~/linear{suffix}"
                post_p = m.get(linear_key, 0)

            node_id_dense = f"PostDense{i + 1}"
            nodes[node_id_dense] = {
                "label": _lbl("Dense", f"{curr_in_sz}→64", post_p),
                "type": "styleMlp",
            }
            edges.append((curr_out, node_id_dense))
            curr_out = node_id_dense
            post_nodes.append(node_id_dense)

            if use_ln:
                idx = pre + i
                suffix_ln = f"_{idx}" if idx > 0 else ""
                ln_p = m.get(f"phi/ForagerGRUNetReLU/~/layer_norm{suffix_ln}", 0)
                node_id_ln = f"PostLN{i + 1}"
                nodes[node_id_ln] = {
                    "label": _lbl("LayerNorm", "64", ln_p),
                    "type": "styleNorm",
                }
                edges.append((curr_out, node_id_ln))
                curr_out = node_id_ln
                post_nodes.append(node_id_ln)

            node_id_act = f"PostAct{i + 1}"
            nodes[node_id_act] = {"label": "ReLU", "type": "styleNorm"}
            edges.append((curr_out, node_id_act))
            curr_out = node_id_act
            post_nodes.append(node_id_act)

            curr_in_sz = 64

    nodes["QHead"] = {"label": _lbl("Q-Head", "64→4", qh_p), "type": "styleHead"}
    edges.append((curr_out, "QHead"))

    subgraphs = []
    if pre_nodes:
        subgraphs.append({"id": "PreGRU", "label": "Pre-GRU", "nodes": pre_nodes})
    if post_nodes:
        subgraphs.append({"id": "PostGRU", "label": "Post-GRU", "nodes": post_nodes})

    core_nodes = [
        "Obs",
        "Scalars",
        "Conv",
        "ReLU1",
        "Flat",
        "Concat",
        "GRU",
        "ReLU2",
        "Skip",
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "subgraphs": [
            {
                "id": "ForagerNet",
                "label": "DRQN (ForagerGRUNetReLU)",
                "nodes": core_nodes,
                "subgraphs": subgraphs,
            }
        ],
    }


# ── PPO graph builders ────────────────────────────────────────────────────────


def build_ppo_graph(variant_data):
    m = variant_data["modules"]
    use_rt = variant_data["use_reward_trace"]
    use_ln = variant_data["use_layernorm"]
    obs_flat = variant_data["obs_flat"]
    obs_hid = variant_data["obs_hidden"]
    cat_sz = variant_data["concat_size"]
    d_hid = variant_data["d_hidden"]
    hid = variant_data["hidden_size"]
    scalars_count = variant_data["scalars"]
    fov = variant_data["fov"]

    a_bits = 4
    r_bits = 1
    rt_bits = 1 if use_rt else 0
    h_bits = 4 if variant_data.get("has_hint") else 0

    scalar_label = f"A[{a_bits}], R[{r_bits}]"
    if use_rt:
        scalar_label += f", RT[{rt_bits}]"
    if h_bits > 0:
        scalar_label += f", Hint[{h_bits}]"

    ad1 = m.get("actor_dense1", 0)
    cd1 = m.get("critic_dense1", 0)
    ad2 = m.get("actor_dense2", 0)
    ad3 = m.get("actor_dense3", 0)
    cd2 = m.get("critic_dense2", 0)
    cd3 = m.get("critic_dense3", 0)
    aln1 = m.get("actor_layernorm1", 0)
    cln1 = m.get("critic_layernorm1", 0)
    aln2 = m.get("actor_layernorm2", 0)
    cln2 = m.get("critic_layernorm2", 0)
    ahead = m.get("actor_mean", 0)
    chead = m.get("critic_value", 0)

    nodes = {
        "Obs": {
            "label": _lbl("Observation", f"{fov}×{fov}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, f"[{scalars_count}]"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Flat": {"label": _lbl("Flatten", f"→{obs_flat}"), "type": "styleMlp"},
        # Actor path
        "A_Emb": {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", ad1),
            "type": "styleMlp",
        },
        "A_Concat": {"label": _lbl("Concat", f"→{cat_sz}"), "type": "styleConcat"},
        "A_Dense2": {
            "label": _lbl("Dense", f"{cat_sz}→{d_hid}", ad2),
            "type": "styleMlp",
        },
        "A_Dense3": {
            "label": _lbl("Dense", f"{d_hid}→{hid}", ad3),
            "type": "styleMlp",
        },
        "A_Act3": {"label": "Tanh", "type": "styleNorm"},
        "A_Head": {"label": _lbl("Logits", f"{hid}→4", ahead), "type": "styleHead"},
        # Critic path
        "C_Emb": {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", cd1),
            "type": "styleMlp",
        },
        "C_Concat": {"label": _lbl("Concat", f"→{cat_sz}"), "type": "styleConcat"},
        "C_Dense2": {
            "label": _lbl("Dense", f"{cat_sz}→{d_hid}", cd2),
            "type": "styleMlp",
        },
        "C_Dense3": {
            "label": _lbl("Dense", f"{d_hid}→{hid}", cd3),
            "type": "styleMlp",
        },
        "C_Act3": {"label": "Tanh", "type": "styleNorm"},
        "C_Head": {"label": _lbl("Value", f"{hid}→1", chead), "type": "styleHead"},
    }

    actor_nodes = ["A_Emb", "A_Concat", "A_Dense2", "A_Dense3", "A_Act3", "A_Head"]
    critic_nodes = ["C_Emb", "C_Concat", "C_Dense2", "C_Dense3", "C_Act3", "C_Head"]
    edges = [
        ("Obs", "Flat"),
        ("Flat", "A_Emb"),
        ("Flat", "C_Emb"),
        ("Scalars", "A_Concat"),
        ("Scalars", "C_Concat"),
    ]

    if use_ln:
        nodes["A_LN1"] = {
            "label": _lbl("LayerNorm", f"{obs_hid}", aln1),
            "type": "styleNorm",
        }
        nodes["A_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["A_LN2"] = {
            "label": _lbl("LayerNorm", f"{hid}", aln2),
            "type": "styleNorm",
        }
        nodes["C_LN1"] = {
            "label": _lbl("LayerNorm", f"{obs_hid}", cln1),
            "type": "styleNorm",
        }
        nodes["C_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["C_LN2"] = {
            "label": _lbl("LayerNorm", f"{hid}", cln2),
            "type": "styleNorm",
        }
        edges += [
            ("A_Emb", "A_LN1"),
            ("A_LN1", "A_Act1"),
            ("A_Act1", "A_Concat"),
            ("A_Concat", "A_Dense2"),
            ("A_Dense2", "A_Dense3"),
            ("A_Dense3", "A_LN2"),
            ("A_LN2", "A_Act3"),
            ("A_Act3", "A_Head"),
            ("C_Emb", "C_LN1"),
            ("C_LN1", "C_Act1"),
            ("C_Act1", "C_Concat"),
            ("C_Concat", "C_Dense2"),
            ("C_Dense2", "C_Dense3"),
            ("C_Dense3", "C_LN2"),
            ("C_LN2", "C_Act3"),
            ("C_Act3", "C_Head"),
        ]
        actor_nodes = [
            "A_Emb",
            "A_LN1",
            "A_Act1",
            "A_Concat",
            "A_Dense2",
            "A_Dense3",
            "A_LN2",
            "A_Act3",
            "A_Head",
        ]
        critic_nodes = [
            "C_Emb",
            "C_LN1",
            "C_Act1",
            "C_Concat",
            "C_Dense2",
            "C_Dense3",
            "C_LN2",
            "C_Act3",
            "C_Head",
        ]
    else:
        nodes["A_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        nodes["C_Act1"] = {"label": "Tanh", "type": "styleNorm"}
        edges += [
            ("A_Emb", "A_Act1"),
            ("A_Act1", "A_Concat"),
            ("A_Concat", "A_Dense2"),
            ("A_Dense2", "A_Dense3"),
            ("A_Dense3", "A_Act3"),
            ("A_Act3", "A_Head"),
            ("C_Emb", "C_Act1"),
            ("C_Act1", "C_Concat"),
            ("C_Concat", "C_Dense2"),
            ("C_Dense2", "C_Dense3"),
            ("C_Dense3", "C_Act3"),
            ("C_Act3", "C_Head"),
        ]
        actor_nodes = [
            "A_Emb",
            "A_Act1",
            "A_Concat",
            "A_Dense2",
            "A_Dense3",
            "A_Act3",
            "A_Head",
        ]
        critic_nodes = [
            "C_Emb",
            "C_Act1",
            "C_Concat",
            "C_Dense2",
            "C_Dense3",
            "C_Act3",
            "C_Head",
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "subgraphs": [
            {"id": "Actor", "label": "Actor", "nodes": actor_nodes},
            {"id": "Critic", "label": "Critic", "nodes": critic_nodes},
        ],
        "invisible_edges": [["Actor", "Critic"]],
    }


def build_ppo_rtu_graph(variant_data):
    m = variant_data["modules"]
    use_rt = variant_data["use_reward_trace"]
    use_ln = variant_data["use_layernorm"]
    obs_flat = variant_data["obs_flat"]
    obs_hid = variant_data["obs_hidden"]
    cat_sz = variant_data["concat_size"]
    d_hid = variant_data["d_hidden"]
    hid = variant_data["hidden_size"]
    scalars_count = variant_data["scalars"]
    fov = variant_data["fov"]

    # Detect if we have 1 or 2 stages
    has_rtu2 = any("rtu2" in k for k in m.keys())

    a_bits = 4
    r_bits = 1
    rt_bits = 1 if use_rt else 0
    h_bits = 4 if variant_data.get("has_hint") else 0

    scalar_label = f"A[{a_bits}], R[{r_bits}]"
    if use_rt:
        scalar_label += f", RT[{rt_bits}]"
    if h_bits > 0:
        scalar_label += f", Hint[{h_bits}]"

    # Robust param gathering
    def sum_p(prefix):
        return sum(v for k, v in m.items() if prefix in k)

    ad1 = sum_p("actor_dense1")
    cd1 = sum_p("critic_dense1")
    ad2 = sum_p("actor_dense2")
    cd2 = sum_p("critic_dense2")
    ad3 = sum_p("actor_dense3")
    cd3 = sum_p("critic_dense3")

    aln1 = sum_p("actor_layernorm1")
    cln1 = sum_p("critic_layernorm1")
    aln2 = sum_p("actor_layernorm2")
    cln2 = sum_p("critic_layernorm2")

    ahead = sum_p("actor_mean")
    chead = sum_p("critic_value")

    if has_rtu2:
        artu1 = sum_p("actor_rtu1")
        crtu1 = sum_p("critic_rtu1")
        artu2 = sum_p("actor_rtu2")
        crtu2 = sum_p("critic_rtu2")
    else:
        artu1 = sum_p("actor_rtu")
        crtu1 = sum_p("critic_rtu")
        artu2 = 0
        crtu2 = 0

    # RTU outputs 2×d_hidden, then concat with skip(cat_sz)
    rtu_out = 2 * d_hid
    skip_out = rtu_out + cat_sz

    nodes = {
        "Obs": {
            "label": _lbl("Observation", f"{fov}×{fov}×3"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Scalars": {
            "label": _lbl(scalar_label, f"[{scalars_count}]"),
            "type": "styleInput",
            "shape": "stadium",
        },
        "Flat": {"label": _lbl("Flatten", f"→{obs_flat}"), "type": "styleMlp"},
    }

    edges = [
        ("Obs", "Flat"),
    ]

    def add_path(
        prefix,
        emb_p,
        ln1_p,
        ln2_p,
        rtu1_p,
        d2_p,
        rtu2_p,
        d3_p,
        head_p,
        head_lbl,
        head_out,
    ):
        p_nodes = []
        # Emb
        nodes[f"{prefix}_Emb"] = {
            "label": _lbl("Dense", f"{obs_flat}→{obs_hid}", emb_p),
            "type": "styleMlp",
        }
        p_nodes.append(f"{prefix}_Emb")
        curr = f"{prefix}_Emb"
        if use_ln:
            nodes[f"{prefix}_LN1"] = {
                "label": _lbl("LayerNorm", f"{obs_hid}", ln1_p),
                "type": "styleNorm",
            }
            edges.append((curr, f"{prefix}_LN1"))
            curr = f"{prefix}_LN1"
            p_nodes.append(f"{prefix}_LN1")
        nodes[f"{prefix}_Act1"] = {"label": "tanh", "type": "styleNorm"}
        edges.append((curr, f"{prefix}_Act1"))
        curr = f"{prefix}_Act1"
        p_nodes.append(f"{prefix}_Act1")

        # Concat
        nodes[f"{prefix}_Concat"] = {
            "label": _lbl("Concat", f"→{cat_sz}"),
            "type": "styleConcat",
        }
        edges.append((curr, f"{prefix}_Concat"))
        edges.append(("Scalars", f"{prefix}_Concat"))
        curr = f"{prefix}_Concat"
        p_nodes.append(f"{prefix}_Concat")
        skip_ref = curr

        # RTU 1
        nodes[f"{prefix}_RTU1"] = {
            "label": _lbl("RTU", f"{cat_sz}→2×{d_hid}", rtu1_p),
            "type": "styleRnn",
        }
        edges.append((curr, f"{prefix}_RTU1"))
        p_nodes.append(f"{prefix}_RTU1")

        # Skip 1
        nodes[f"{prefix}_Skip1"] = {
            "label": _lbl("Concat", f"→{skip_out}"),
            "type": "styleConcat",
        }
        edges.append((f"{prefix}_RTU1", f"{prefix}_Skip1"))
        edges.append((skip_ref, f"{prefix}_Skip1"))
        curr = f"{prefix}_Skip1"
        p_nodes.append(f"{prefix}_Skip1")

        if has_rtu2:
            # Dense 2
            nodes[f"{prefix}_Dense2"] = {
                "label": _lbl("Dense", f"{skip_out}→{obs_hid}", d2_p),
                "type": "styleMlp",
            }
            nodes[f"{prefix}_Act2"] = {"label": "tanh", "type": "styleNorm"}
            edges.append((curr, f"{prefix}_Dense2"))
            edges.append((f"{prefix}_Dense2", f"{prefix}_Act2"))
            curr = f"{prefix}_Act2"
            skip_ref = curr
            p_nodes.append(f"{prefix}_Dense2")
            p_nodes.append(f"{prefix}_Act2")

            # RTU 2
            nodes[f"{prefix}_RTU2"] = {
                "label": _lbl("RTU 2", f"{obs_hid}→2×{d_hid}", rtu2_p),
                "type": "styleRnn",
            }
            edges.append((curr, f"{prefix}_RTU2"))
            p_nodes.append(f"{prefix}_RTU2")

            # Skip 2
            nodes[f"{prefix}_Skip2"] = {
                "label": _lbl("Concat", f"→{skip_out}"),
                "type": "styleConcat",
            }
            edges.append((f"{prefix}_RTU2", f"{prefix}_Skip2"))
            edges.append((skip_ref, f"{prefix}_Skip2"))
            curr = f"{prefix}_Skip2"
            p_nodes.append(f"{prefix}_Skip2")

            # Final Dense
            nodes[f"{prefix}_Dense3"] = {
                "label": _lbl("Dense", f"{skip_out}→{hid}", d3_p),
                "type": "styleMlp",
            }
            curr = f"{prefix}_Dense3"
            p_nodes.append(f"{prefix}_Dense3")
        else:
            # Final Dense
            nodes[f"{prefix}_Dense2"] = {
                "label": _lbl("Dense", f"{skip_out}→{hid}", d2_p),
                "type": "styleMlp",
            }
            edges.append((curr, f"{prefix}_Dense2"))
            curr = f"{prefix}_Dense2"
            p_nodes.append(f"{prefix}_Dense2")

        if use_ln:
            nodes[f"{prefix}_LN2"] = {
                "label": _lbl("LayerNorm", f"{hid}", ln2_p),
                "type": "styleNorm",
            }
            edges.append((curr, f"{prefix}_LN2"))
            curr = f"{prefix}_LN2"
            p_nodes.append(f"{prefix}_LN2")

        nodes[f"{prefix}_ActF"] = {"label": "tanh", "type": "styleNorm"}
        nodes[f"{prefix}_Head"] = {
            "label": _lbl(head_lbl, f"{hid}→{head_out}", head_p),
            "type": "styleHead",
        }
        edges.append((curr, f"{prefix}_ActF"))
        edges.append((f"{prefix}_ActF", f"{prefix}_Head"))
        p_nodes.append(f"{prefix}_ActF")
        p_nodes.append(f"{prefix}_Head")
        return p_nodes

    actor_n = add_path("A", ad1, aln1, aln2, artu1, ad2, artu2, ad3, ahead, "Logits", 4)
    critic_n = add_path("C", cd1, cln1, cln2, crtu1, cd2, crtu2, cd3, chead, "Value", 1)

    edges.insert(0, ("Flat", "A_Emb"))
    edges.insert(0, ("Flat", "C_Emb"))

    return {
        "nodes": nodes,
        "edges": edges,
        "subgraphs": [
            {"id": "Actor", "label": "Actor Path", "nodes": actor_n},
            {"id": "Critic", "label": "Critic Path", "nodes": critic_n},
        ],
        "invisible_edges": [["A_Head", "C_Head"]],
    }


# ── Param extraction ──────────────────────────────────────────────────────────


def get_dqn_data(config_name):
    path = f"{BASE}/{config_name}.json"
    exp = load_exp(path)
    hypers = exp.get_hypers(0)
    hypers["total_steps"] = exp.total_steps
    env_info = hypers.get("environment", {})
    env_id = env_info.get("env_id", "ForagaxBig-v4")
    env_params = {k: v for k, v in env_info.items() if k != "env_id"}

    env = make_env(env_id=env_id, **env_params)
    obs_space = env.observation_space(env.default_params)
    hints_size = obs_space.spaces["hint"].shape[0] if "hint" in obs_space.spaces else 0

    agent_name = exp.agent
    rep = hypers["representation"]

    # Use raw JSON for scalar features to avoid PyExpUtils list expansion
    with open(path, "r") as f:
        config_raw = json.load(f)
    rep_raw = config_raw.get("metaParameters", {}).get("representation", {})
    scalar_features = rep_raw.get("scalar_features", ["last_action", "last_reward"])
    if (
        isinstance(scalar_features, list)
        and len(scalar_features) == 1
        and isinstance(scalar_features[0], list)
    ):
        scalar_features = scalar_features[0]

    use_rt = "reward_trace" in scalar_features
    # Redo NNAgent.py calculation
    scalars_size = 0
    if "last_action" in scalar_features:
        scalars_size += ACTIONS
    if "last_reward" in scalar_features:
        scalars_size += 1
    if "reward_trace" in scalar_features:
        scalars_size += 1
    if hints_size > 0:  # Force hint for Big-v4
        scalars_size += hints_size

    fov = env_params.get("aperture_size", FOV)
    obs_shape = (fov, fov, 3)

    rep["scalars"] = scalars_size
    Agent = getAgent(agent_name)
    agent = Agent(obs_shape, ACTIONS, hypers, Collector(), 0)
    params = agent.builder.getParams()

    conv_out = 16 * fov * fov
    cat_out = conv_out + scalars_size

    d = {
        "variant": config_name,
        "fov": fov,
        "total_params": sum(v.size for v in jax.tree_util.tree_leaves(params)),
        "obs_shape": f"{fov}×{fov}×3",
        "obs_shape_tuple": list(obs_shape),
        "scalars": scalars_size,
        "has_rt": use_rt,
        "use_layernorm": rep.get("use_layernorm", False),
        "conv_out": conv_out,
        "concat_size": cat_out,
        "modules": params_by_module(params),
        "has_hint": hints_size > 0,
        "hints_size": hints_size,
    }
    is_drqn = config_name.startswith("DRQN")
    if is_drqn:
        d["pre_core_layers"] = rep.get(
            "pre_core_layers", rep.get("pre_gru_layers", 0)
        )
        d["post_core_layers"] = rep.get(
            "post_core_layers", rep.get("post_gru_layers", 0)
        )

    d["graph"] = build_drqn_graph(d) if is_drqn else build_dqn_graph(d)
    return d


def get_ppo_data(config_name):
    path = f"{BASE}/{config_name}.json"
    exp = load_exp(path)
    hypers = exp.get_hypers(0)
    env_info = hypers.get("environment", {})
    env_id = env_info.get("env_id", "ForagaxBig-v4")
    env_params = {k: v for k, v in env_info.items() if k != "env_id"}

    env = make_env(env_id=env_id, **env_params)
    obs_space = env.observation_space(env.default_params)
    hints_size = obs_space.spaces["hint"].shape[0] if "hint" in obs_space.spaces else 0

    agent_name = exp.agent

    # Use raw JSON to avoid PyExpUtils sweep expansion
    with open(path, "r") as f:
        config_raw = json.load(f)
    rep_raw = config_raw.get("metaParameters", {}).get("representation", {})
    use_rt = rep_raw.get("use_reward_trace", False)
    use_ln = rep_raw.get("use_layernorm", False)
    d_hidden = rep_raw.get("d_hidden", 192)
    hidden_size = rep_raw.get("hidden", 64)

    # NNAgent/PPORegistry logic: A[4] + R[1] + RT[1 if use_rt] + Hint[4 if hint]
    if "ForagaxBig-v4" in env_id:
        if "representation" not in hypers:
            hypers["representation"] = {}
        if "scalar_features" not in hypers["representation"]:
            hypers["representation"]["scalar_features"] = ["last_action", "last_reward"]
        if "hint" not in hypers["representation"]["scalar_features"]:
            hypers["representation"]["scalar_features"].append("hint")

    scalars_count = 5 + (1 if use_rt else 0) + hints_size

    fov = env_params.get("aperture_size", FOV)
    obs_shape = (fov, fov, 3)
    obs_flat = fov * fov * 3
    obs_hid = hidden_size - scalars_count  # Reduced embedding logic

    batch = 1
    obs = (
        jnp.zeros((batch, *obs_shape)),  # image
        jnp.zeros((batch, ACTIONS)),  # last_action (one-hot)
        jnp.zeros((batch, 1 + hints_size)),  # last_reward + hint
        jnp.zeros((batch, 1)),  # sine
        jnp.zeros((batch, 1)),  # cosine
        jnp.zeros((batch, 1)),  # reward_trace
    )

    Agent = getPPOAgent(agent_name)
    model = Agent(
        action_dim=ACTIONS,
        d_hidden=d_hidden,
        hidden_size=hidden_size,
        use_reward_trace=use_rt,
        use_layernorm=use_ln,
    )

    hstate = model.initialize_memory(batch, d_hidden, hidden_size)
    key = jax.random.PRNGKey(0)
    params = model.init(key, hstate, obs)

    def get_flat_params(params):
        res = {}

        def _walk(d, prefix=""):
            for k, v in d.items():
                kp = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict) or hasattr(v, "items"):
                    _walk(v, kp)
                else:
                    mod = kp
                    if mod.startswith("params/"):
                        mod = mod[len("params/") :]
                    mod = mod.rsplit("/", 1)[0] if "/" in mod else mod
                    res[mod] = res.get(mod, 0) + int(v.size)

        _walk(params)
        return res

    params_flat = get_flat_params(params)

    d = {
        "variant": config_name,
        "fov": fov,
        "total_params": sum(v.size for v in jax.tree_util.tree_leaves(params)),
        "obs_shape": f"{fov}×{fov}×3",
        "obs_shape_tuple": list(obs_shape),
        "scalars": scalars_count,
        "obs_flat": obs_flat,
        "obs_hidden": obs_hid,
        "concat_size": obs_hid + scalars_count,
        "d_hidden": d_hidden,
        "hidden_size": hidden_size,
        "use_reward_trace": use_rt,
        "use_layernorm": use_ln,
        "modules": params_flat,
        "has_hint": hints_size > 0,
        "hints_size": hints_size,
    }

    if "RTU" in agent_name:
        d["graph"] = build_ppo_rtu_graph(d)
    else:
        d["graph"] = build_ppo_graph(d)
    return d


# ── Build all variants ────────────────────────────────────────────────────────

META = [
    (
        "DQN_LN_RT",
        lambda n: get_dqn_data(n),
        build_dqn_graph,
        "DQN_LN_RT",
        "DQN with LayerNorm and Reward Trace.",
    ),
    (
        "DQN_LN",
        lambda n: get_dqn_data(n),
        build_dqn_graph,
        "DQN_LN",
        "DQN with LayerNorm.",
    ),
    (
        "DQN",
        lambda n: get_dqn_data(n),
        build_dqn_graph,
        "DQN",
        "Baseline DQN without LayerNorm.",
    ),
    (
        "DRQN_LN_0_2",
        get_dqn_data,
        build_drqn_graph,
        "DRQN_LN_0_2",
        "Recurrent DQN with LayerNorm, 0 pre-GRU layers.",
    ),
    (
        "DRQN_LN_1_1",
        get_dqn_data,
        build_drqn_graph,
        "DRQN_LN_1_1",
        "Recurrent DQN with LayerNorm, 1 pre-GRU layer.",
    ),
    (
        "DRQN_0_2",
        get_dqn_data,
        build_drqn_graph,
        "DRQN_0_2",
        "Recurrent DQN without LayerNorm, 0 pre-GRU layers.",
    ),
    (
        "DRQN_1_1",
        get_dqn_data,
        build_drqn_graph,
        "DRQN_1_1",
        "Recurrent DQN without LayerNorm, 1 pre-GRU layer.",
    ),
    (
        "PPO_LN_RT_128",
        get_ppo_data,
        build_ppo_graph,
        "PPO_LN_RT_128",
        "PPO with LayerNorm and Reward Trace.",
    ),
    ("PPO_LN_128", get_ppo_data, build_ppo_graph, "PPO_LN_128", "PPO with LayerNorm."),
    (
        "PPO_128",
        get_ppo_data,
        build_ppo_graph,
        "PPO_128",
        "Baseline PPO without LayerNorm.",
    ),
    (
        "PPO-RTU_LN_128_512",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_LN_128_512",
        "RTU PPO with LayerNorm, d_hidden=512.",
    ),
    (
        "PPO-RTU_LN_128",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_LN_128",
        "RTU PPO with LayerNorm, d_hidden=192.",
    ),
    (
        "PPO-RTU_128_512",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_128",
        "RTU PPO without LayerNorm, d_hidden=512.",
    ),
    (
        "PPO-RTU_128",
        get_ppo_data,
        build_ppo_rtu_graph,
        "PPO-RTU_128",
        "RTU PPO without LayerNorm, d_hidden=192.",
    ),
]

arch_data = {}
for config_name, data_fn, graph_fn, title, desc in META:
    print(f"Processing {config_name}...")
    try:
        data = data_fn(config_name)
        data["title"] = title
        data["description"] = desc
        graph = graph_fn(data)
        data["graph"] = graph
        data["mermaid_source"] = graph_to_mermaid(graph)
        arch_data[config_name] = data
    except Exception as e:
        import traceback

        print(f"  ERROR {config_name}: {e}")
        traceback.print_exc()

out_path = "public/arch_data.json"
with open(out_path, "w") as f:
    json.dump(arch_data, f, indent=2)

print(f"\nExported {len(arch_data)} variants → {out_path}")
print(f"\n{'Variant':<25}  {'Params':>12}")
print("-" * 40)
for name, d in arch_data.items():
    print(f"  {name:<23}  {d['total_params']:>12,}")
