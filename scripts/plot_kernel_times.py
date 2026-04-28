import argparse
import json
import re
import matplotlib.pyplot as plt


def _find_matching_angle(s, start_idx):
    depth = 0
    for i in range(start_idx, len(s)):
        c = s[i]
        if c == '<':
            depth += 1
        elif c == '>':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(s, sep=','):
    out, cur = [], []
    angle = paren = bracket = brace = 0

    for c in s:
        if c == '<':
            angle += 1
        elif c == '>':
            angle -= 1
        elif c == '(':
            paren += 1
        elif c == ')':
            paren -= 1
        elif c == '[':
            bracket += 1
        elif c == ']':
            bracket -= 1
        elif c == '{':
            brace += 1
        elif c == '}':
            brace -= 1

        if c == sep and angle == 0 and paren == 0 and bracket == 0 and brace == 0:
            out.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(c)

    if cur:
        out.append(''.join(cur).strip())
    return out


def _extract_nv_dl_callable(name):
    tag = '__nv_dl_tag<'
    i = name.find(tag)
    if i < 0:
        return None

    start = i + len(tag) - 1
    end = _find_matching_angle(name, start)
    if end < 0:
        return None

    inside = name[start + 1:end]
    parts = _split_top_level(inside, ',')
    if len(parts) < 2:
        return None

    cand = parts[1].strip()
    if cand.startswith('&'):
        cand = cand[1:]
    return cand


def _normalize_scream_internal(s):
    m = re.search(
        r'^scream::_GLOBAL__N__[^:]*?_eamxx_mam_([a-z0-9_]+)_process_interface_cpp_[^:]*::([A-Za-z0-9_]+)$',
        s,
    )
    if not m:
        return s

    process_key, func = m.group(1), m.group(2)
    alias = {
        'dry_deposition': 'dry_deposition',
        'aci': 'aci',
    }.get(process_key, process_key)

    return f'{alias}::{func}'


def short_name(name):
    s = str(name).strip()

    if s.startswith('&') and '::' in s:
        return _normalize_scream_internal(s[1:])

    callable_name = _extract_nv_dl_callable(s)
    if callable_name:
        return _normalize_scream_internal(callable_name)

    if s.startswith('CombinedFunctorReducer<'):
        inner = _extract_nv_dl_callable(s)
        if inner:
            return f"CombinedFunctorReducer::{_normalize_scream_internal(inner)}"

    for prefix in (
        'Kokkos::View::initialization',
        'Kokkos::View::allocation',
        'Kokkos::DeepCopy',
    ):
        if s.startswith(prefix):
            return prefix

    return _normalize_scream_internal(s)


def plot_kernel_times(json_file, kernel_names=None, output="kernel_times.png", top_n=None, title="Kernel Total Time"):
    with open(json_file) as f:
        data = json.load(f)

    kernel_map = {
        short_name(entry["kernel-name"]): entry["total-time"]
        for entry in data["kernel-data"]
    }
    # print(kernel_map)

    # If no kernel names specified, use top N by total-time
    if kernel_names is None:
        if top_n is None:
            top_n = 10
        sorted_kernels = sorted(kernel_map.items(), key=lambda x: x[1], reverse=True)
        names = [k for k, v in sorted_kernels[:top_n]]
        times = [v for k, v in sorted_kernels[:top_n]]
    else:
        names = []
        times = []
        missing = []
        for name in kernel_names:
            if name in kernel_map:
                names.append(name)
                times.append(kernel_map[name])
            else:
                missing.append(name)

        if missing:
            print(f"Warning: kernels not found in data: {missing}")

        if not names:
            print("No matching kernels found. Exiting.")
            return

    # Use short labels for readability
    labels = [short_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bars = ax.bar(range(len(names)), times)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Total Time (s)")
    ax.set_title(title)

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bar plot total-time for selected kernels from a Kokkos JSON profile."
    )
    parser.add_argument("json_file", help="Path to the Kokkos JSON profile file.")
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="+",
        required=False,
        metavar="KERNEL_NAME",
        help="List of kernel names to plot. If not specified, plots top 10 kernels by total-time.",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=10,
        help="Number of top kernels to plot when --kernels is not specified (default: 10).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="kernel_times.png",
        help="Output image file (default: kernel_times.png).",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="Kernel Total Time",
        help="Title of the plot (default: 'Kernel Total Time').",
    )
    args = parser.parse_args()
    plot_kernel_times(args.json_file, kernel_names=args.kernels, output=args.output, top_n=args.top_n, title=args.title)
