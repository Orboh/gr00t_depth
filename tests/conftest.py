"""Tests run against NVIDIA's real GR00T N1.7 action head, built with a tiny
config so they run on CPU. Point GR00T_REPO at a checkout of
NVIDIA/Isaac-GR00T if `gr00t` isn't already importable."""
import os
import sys

repo = os.environ.get("GR00T_REPO")
if repo and repo not in sys.path:
    sys.path.insert(0, repo)
