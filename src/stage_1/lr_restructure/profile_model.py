import argparse

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

from stage_1.lr_restructure.lr_sr_encoder import LRRestructureEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-atoms", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-master-nodes", type=int, default=4)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--box", type=float, default=20.0)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--trace-out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--compile-mode", type=str, default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune"],
        help="reduce-overhead captures a CUDA graph to eliminate per-kernel launch overhead -- "
             "the mode that actually targets the symptom this script measures.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}", flush=True)

    torch.manual_seed(0)
    model = LRRestructureEncoder(
        num_master_nodes=args.num_master_nodes, feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim, n_max=args.n_max, num_outputs=1,
        num_layers=args.num_layers, edge_cutoff=5.0, sigma=args.sigma,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model = model
    if args.torch_compile:
        # The remaining recurring graph break in this model is pair_mask.nonzero(as_tuple=True)
        # (model.py, building the edge list) -- its output shape is genuinely data-dependent, so
        # ask Dynamo to trace it symbolically instead of hard-breaking the graph there.
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        compile_kwargs = {}
        if args.compile_mode == "reduce-overhead":
            # CUDA graph capture wants a fixed shape per graph, not symbolic dims -- dynamic=True
            # (used for the other modes so a fixed --n-atoms/--batch-size doesn't recompile across
            # runs) actively works against that, so leave shape specialization on here.
            compile_kwargs["mode"] = args.compile_mode
        else:
            compile_kwargs["dynamic"] = True
            if args.compile_mode:
                compile_kwargs["mode"] = args.compile_mode
        train_model = torch.compile(model, **compile_kwargs)

    B, N = args.batch_size, args.n_atoms
    charges = torch.sign(torch.randn(B, N, device=device))
    positions = torch.rand(B, N, 3, device=device) * args.box
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    V = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1) * args.box
    target = torch.randn(B, 1, device=device)

    def step():
        opt.zero_grad()
        with record_function("lr_forward"):
            pred = train_model(charges, positions, mask, V)
        with record_function("lr_loss_backward"):
            loss = F.mse_loss(pred, target)
            loss.backward()
        opt.step()
        return loss

    print(f"Warming up ({args.warmup_steps} steps)...", flush=True)
    for _ in range(args.warmup_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling ({args.profile_steps} steps)...", flush=True)
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.profile_steps):
            with record_function("lr_train_step"):
                step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

    print("\n" + "=" * 100)
    print(f"TOP OPS BY {sort_key.upper()} (time-bound hotspots)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))

    print("\n" + "=" * 100)
    print("TOP OPS BY CALL COUNT (kernel-launch-bound hotspots -- high count + tiny per-call time")
    print("means launch overhead, not compute, is the bottleneck)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="count", row_limit=args.row_limit))

    total_calls = sum(e.count for e in prof.key_averages())
    print(f"\nTotal recorded op invocations across {args.profile_steps} step(s): {total_calls}")
    print(f"-> ~{total_calls / args.profile_steps:.0f} op invocations per training step")

    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"\nChrome trace written to {args.trace_out} (open in chrome://tracing or "
              f"https://ui.perfetto.dev for a visual, per-launch timeline).")


if __name__ == "__main__":
    main()
