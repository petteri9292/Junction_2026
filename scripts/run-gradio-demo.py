"""Local/private Gradio launcher; the same setup is in notebook10."""

import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wikilens.gradio_demo import build_demo, launch_demo

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--live", action="store_true")
parser.add_argument("--remote-url")
args = parser.parse_args()
demo, controller = build_demo(
    "Live search" if args.live else "Saved run (offline)", remote_url=args.remote_url
)
launch_demo(demo, port=args.port, inline=False)
print("READY: private localhost Gradio UI", flush=True)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    demo.close()
    controller.close()
