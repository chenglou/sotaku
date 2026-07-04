# Benchmark the ES generation loop: sequential-eager (production today) against
# batched-population and compiled variants, on a real trained model, measuring both
# wall time and fitness fidelity (same perturbation seeds must give the same cell
# counts, within batching-order rounding).
#
# Run on Modal:  modal run --detach modal_run.py --exp es.bench_es
# Reads the seed model from output_dir (the Modal outputs volume).

import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

from iters.exp_baseline_lr2e3 import ROPE_COS, ROPE_SIN, SudokuTransformer, encode_puzzles

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "bench_es_checkpoint_step"
total_steps = 1
eval_every = 1
log_name = "bench_es.log"

SEED_MODEL = "model_viridian_canonical_final50k.pt"
POPULATION = 32           # 16 antithetic pairs
FITNESS_PUZZLES = 384
FITNESS_ITERS = 1024
SIGMA = 3e-4
COMPILE_CHUNK = 32        # iterations per compiled block (proven shape in exp_testbed_burnin128)


def make_noise(param_shapes, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [torch.randn(shape, generator=gen, device=device, dtype=torch.float32) for shape in param_shapes]


def run_iterations_eager(model, x, n_iters, rope_cos, rope_sin):
    h_prev = model.initial_encoder(x)
    preds = torch.zeros(x.size(0), 81, 9, device=x.device)
    for _ in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        preds = F.softmax(model.output_head(h), dim=-1)
    return model.output_head(h_prev)


def count_cells(final_argmax, targets, empty_mask):
    return ((final_argmax == targets) & empty_mask).sum(dim=(-2, -1))


def train(output_dir="."):
    device = torch.device("cuda")
    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    rows = dataset[2_700_000:2_700_000 + FITNESS_PUZZLES]
    x = encode_puzzles(rows["question"]).to(device)
    targets = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in rows["answer"]], device=device)
    empty = torch.tensor([[p[j] == '.' for j in range(81)] for p in rows["question"]], device=device)

    model = SudokuTransformer().to(device)
    state = torch.load(os.path.join(output_dir, SEED_MODEL), map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    params = list(model.parameters())
    param_shapes = [p.shape for p in params]
    rope_cos = ROPE_COS.to(device)
    rope_sin = ROPE_SIN.to(device)

    seeds = list(range(1000, 1000 + POPULATION))
    results = {}

    # ---------- v0: sequential eager, 256-chunks — production today ----------
    def v0():
        counts = []
        snapshot = [p.detach().clone() for p in params]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for seed in seeds:
                noise = make_noise(param_shapes, seed, device)
                for p, z in zip(params, noise):
                    p.add_(z, alpha=SIGMA)
                total = 0
                for start in range(0, x.size(0), 256):
                    final = run_iterations_eager(model, x[start:start + 256], FITNESS_ITERS, rope_cos, rope_sin).argmax(dim=-1)
                    total += int(count_cells(final, targets[start:start + 256], empty[start:start + 256]).item())
                counts.append(total)
                for p, s in zip(params, snapshot):
                    p.copy_(s)
        return counts

    # ---------- v0b: sequential eager, one 384 chunk ----------
    def v0b():
        counts = []
        snapshot = [p.detach().clone() for p in params]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for seed in seeds:
                noise = make_noise(param_shapes, seed, device)
                for p, z in zip(params, noise):
                    p.add_(z, alpha=SIGMA)
                final = run_iterations_eager(model, x, FITNESS_ITERS, rope_cos, rope_sin).argmax(dim=-1)
                counts.append(int(count_cells(final, targets, empty).item()))
                for p, s in zip(params, snapshot):
                    p.copy_(s)
        return counts

    # ---------- v1: sequential, compiled 32-iteration block ----------
    def iter_block(h_prev, preds, xb):
        for _ in range(COMPILE_CHUNK):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            preds = F.softmax(model.output_head(h), dim=-1)
        return h_prev, preds

    compiled_block = torch.compile(iter_block, dynamic=False)

    def v1():
        counts = []
        snapshot = [p.detach().clone() for p in params]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for seed in seeds:
                noise = make_noise(param_shapes, seed, device)
                for p, z in zip(params, noise):
                    p.add_(z, alpha=SIGMA)
                h_prev = model.initial_encoder(x)
                preds = torch.zeros(x.size(0), 81, 9, device=device)
                for _ in range(FITNESS_ITERS // COMPILE_CHUNK):
                    h_prev, preds = compiled_block(h_prev, preds, x)
                final = model.output_head(h_prev).argmax(dim=-1)
                counts.append(int(count_cells(final, targets, empty).item()))
                for p, s in zip(params, snapshot):
                    p.copy_(s)
        return counts

    # ---------- v2: batched population via vmap ----------
    from torch.func import functional_call, stack_module_state, vmap

    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    param_names = [name for name, _ in model.named_parameters()]

    def build_stacked():
        stacked = {}
        noises = {name: [] for name in param_names}
        for seed in seeds:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            for name, p in model.named_parameters():
                noises[name].append(torch.randn(p.shape, generator=gen, device=device, dtype=torch.float32))
        for k, v in base_state.items():
            if k in noises:
                z = torch.stack(noises[k])
                stacked[k] = v.unsqueeze(0) + SIGMA * z
            else:
                stacked[k] = v.unsqueeze(0).expand(POPULATION, *v.shape).contiguous()
        return stacked

    def batched_iterations(stacked, xb, n_iters):
        # vmap over a function that runs the whole refinement loop for one member,
        # calling each submodule functionally with that member's slice of the weights.
        def loop(member_state):
            h_prev = functional_call(model.initial_encoder, {k[len('initial_encoder.'):]: v for k, v in member_state.items() if k.startswith('initial_encoder.')}, (xb,))
            preds = torch.zeros(xb.size(0), 81, 9, device=xb.device)
            pred_proj_state = {k[len('pred_proj.'):]: v for k, v in member_state.items() if k.startswith('pred_proj.')}
            head_state = {k[len('output_head.'):]: v for k, v in member_state.items() if k.startswith('output_head.')}
            layer_states = []
            for li, layer in enumerate(model.layers):
                prefix = f'layers.{li}.'
                layer_states.append({k[len(prefix):]: v for k, v in member_state.items() if k.startswith(prefix)})
            for _ in range(n_iters):
                h = h_prev + functional_call(model.pred_proj, pred_proj_state, (preds,))
                for layer, lstate in zip(model.layers, layer_states):
                    h = functional_call(layer, lstate, (h, rope_cos, rope_sin))
                h_prev = h
                preds = F.softmax(functional_call(model.output_head, head_state, (h,)), dim=-1)
            return functional_call(model.output_head, head_state, (h_prev,))
        return vmap(loop)(stacked)

    def v2():
        stacked = build_stacked()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            final = batched_iterations(stacked, x, FITNESS_ITERS).argmax(dim=-1)
            return count_cells(final, targets, empty).tolist()

    # ---------- v3: batched + compiled block ----------
    def v3():
        stacked = build_stacked()

        def loop_block(h_prev, preds, member_state):
            pred_proj_state = {k[len('pred_proj.'):]: v for k, v in member_state.items() if k.startswith('pred_proj.')}
            head_state = {k[len('output_head.'):]: v for k, v in member_state.items() if k.startswith('output_head.')}
            layer_states = []
            for li, layer in enumerate(model.layers):
                prefix = f'layers.{li}.'
                layer_states.append({k[len(prefix):]: v for k, v in member_state.items() if k.startswith(prefix)})
            for _ in range(COMPILE_CHUNK):
                h = h_prev + functional_call(model.pred_proj, pred_proj_state, (preds,))
                for layer, lstate in zip(model.layers, layer_states):
                    h = functional_call(layer, lstate, (h, rope_cos, rope_sin))
                h_prev = h
                preds = F.softmax(functional_call(model.output_head, head_state, (h,)), dim=-1)
            return h_prev, preds

        batched_block = torch.compile(vmap(loop_block), dynamic=False)

        def init_member(member_state):
            init_state = {k[len('initial_encoder.'):]: v for k, v in member_state.items() if k.startswith('initial_encoder.')}
            return functional_call(model.initial_encoder, init_state, (x,))

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            h_prev = vmap(init_member)(stacked)
            preds = torch.zeros(POPULATION, x.size(0), 81, 9, device=device)
            for _ in range(FITNESS_ITERS // COMPILE_CHUNK):
                h_prev, preds = batched_block(h_prev, preds, stacked)

            def head(member_state, h):
                head_state = {k[len('output_head.'):]: v for k, v in member_state.items() if k.startswith('output_head.')}
                return functional_call(model.output_head, head_state, (h,))

            final = vmap(head)(stacked, h_prev).argmax(dim=-1)
            return count_cells(final, targets, empty).tolist()

    benches = [("v0 sequential eager 256-chunk (production)", v0),
               ("v0b sequential eager full-batch", v0b),
               ("v1 sequential + compiled 32-iter block", v1),
               ("v2 batched population (vmap, eager)", v2),
               ("v3 batched + compiled block", v3)]

    for name, fn in benches:
        try:
            t0 = time.time()
            counts = fn()   # first call includes compile time where applicable
            t_first = time.time() - t0
            t0 = time.time()
            counts2 = fn()
            t_steady = time.time() - t0
            ref = results.get("v0 sequential eager 256-chunk (production)")
            fidelity = ""
            if ref is not None:
                diffs = [abs(a - b) for a, b in zip(counts2, ref)]
                fidelity = f" | vs v0: max diff {max(diffs)} cells, mean {sum(diffs)/len(diffs):.1f}"
            results[name] = counts2
            log(f"{name}: first {t_first:.1f}s, steady {t_steady:.1f}s for {POPULATION} members{fidelity}")
        except Exception as e:
            log(f"{name}: FAILED — {type(e).__name__}: {e}")

    log_file.close()
    return {}


if __name__ == "__main__":
    train()
