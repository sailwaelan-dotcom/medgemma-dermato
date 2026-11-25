"""Supervised 1-click CLI launcher (`medgemma-dermato serve`) for offline Edge Cabinet App."""

import argparse
import atexit
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn


def _find_llama_server(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates = [
        Path("tools/llama-cpp/llama-server.exe"),
        Path("tools/llama-cpp/llama-server"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    on_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return Path(on_path) if on_path else None


def _find_best_gguf(
    explicit_gguf: Path | None,
    explicit_mmproj: Path | None,
) -> tuple[Path | None, Path | None]:
    if explicit_gguf and explicit_mmproj:
        return explicit_gguf, explicit_mmproj

    # Prefer v2 only if explicitly promoted via env or present after validation
    search_dirs = [Path("checkpoints/gguf"), Path("checkpoints/gguf_v2")]
    for d in search_dirs:
        if not d.exists():
            continue
        ggufs = sorted(d.glob("*q8_0*.gguf"))
        mmprojs = sorted(d.glob("*mmproj*.gguf"))
        if ggufs and mmprojs:
            return explicit_gguf or ggufs[0], explicit_mmproj or mmprojs[0]
    return explicit_gguf, explicit_mmproj


def _wait_health(url: str, timeout_s: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(f"{url}/health").status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.75)
    return False


def serve_command(args: argparse.Namespace) -> int:
    """Start local llama-server (if available and enabled) + FastAPI Edge UI cleanly."""
    vlm_proc: subprocess.Popen[bytes] | None = None

    def _cleanup() -> None:
        nonlocal vlm_proc
        if vlm_proc is not None and vlm_proc.poll() is None:
            print("[launcher] Stopping supervised llama-server...", flush=True)
            vlm_proc.terminate()
            try:
                vlm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vlm_proc.kill()
            vlm_proc = None

    atexit.register(_cleanup)

    if not args.no_vlm:
        server_bin = _find_llama_server(args.llama_server)
        gguf_path, mmproj_path = _find_best_gguf(args.gguf, args.mmproj)
        if server_bin and gguf_path and mmproj_path and gguf_path.exists() and mmproj_path.exists():
            vlm_url = f"http://127.0.0.1:{args.vlm_port}"
            os.environ["LLAMA_SERVER_URL"] = vlm_url
            cmd = [
                str(server_bin),
                "-m",
                str(gguf_path),
                "--mmproj",
                str(mmproj_path),
                "-ngl",
                "0" if args.cpu_only else "99",
                "-c",
                "4096",
                "--parallel",
                "1",
                "--cache-prompt",
                "--offline",
                "--port",
                str(args.vlm_port),
            ]
            print(
                f"[launcher] Starting System 2 GGUF ({gguf_path.name}) on port {args.vlm_port}...",
                flush=True,
            )
            vlm_proc = subprocess.Popen(cmd)
            if _wait_health(vlm_url, timeout_s=60.0):
                print(f"[launcher] System 2 llama-server ready at {vlm_url}", flush=True)
            else:
                print(
                    "[launcher] Warning: llama-server did not answer /health in 60s; "
                    "Edge UI will start in System 1 degraded mode.",
                    flush=True,
                )
        else:
            print(
                "[launcher] System 2 GGUF or llama-server binary not found; "
                "starting in System 1 ONNX degraded mode.",
                flush=True,
            )

    app_url = f"http://{args.host}:{args.port}"
    if args.open:
        webbrowser.open(app_url)

    cwd_str = str(Path.cwd())
    if cwd_str not in sys.path:
        sys.path.insert(0, cwd_str)

    try:
        uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")
    finally:
        _cleanup()
    return 0


DEFAULT_SYS1_REVISION = "99707777873c87c1b73244273b05b47d60ce27cd"


def download_system1_command(args: argparse.Namespace) -> int:
    """Download public CC-0/CC-BY System 1 ONNX weights from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    target_dir = Path(args.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("system1_convnext_tiny_fp32.onnx", "system1_calibration.json"):
        print(
            f"[download-system1] Downloading {fname} from {args.repo_id}@{args.revision[:10]}...",
            flush=True,
        )
        hf_hub_download(
            repo_id=args.repo_id,
            filename=fname,
            revision=args.revision,
            local_dir=str(target_dir),
        )
    print(f"[download-system1] Ready in {target_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medgemma-dermato", description="MedGemma Edge CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    srv = sub.add_parser("serve", help="Start supervised llama-server + offline FastAPI UI.")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--vlm-port", type=int, default=8080)
    srv.add_argument("--gguf", type=Path, default=None)
    srv.add_argument("--mmproj", type=Path, default=None)
    srv.add_argument("--llama-server", type=Path, default=None)
    srv.add_argument("--no-vlm", action="store_true")
    srv.add_argument("--cpu-only", action="store_true")
    srv.add_argument("--open", action="store_true")

    dl = sub.add_parser(
        "download-system1",
        help="Download public CC-0/CC-BY System 1 ConvNeXt-Tiny ONNX weights (106 MB).",
    )
    dl.add_argument("--repo-id", default="SChaf09/medgemma-dermato-system1-onnx")
    dl.add_argument("--revision", default=DEFAULT_SYS1_REVISION)
    dl.add_argument("--output-dir", type=Path, default=Path("checkpoints/system1_v1"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve_command(args)
    if args.command == "download-system1":
        return download_system1_command(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
