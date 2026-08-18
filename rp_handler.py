"""RunPod serverless entrypoint.

Deliberately tiny and named to match RunPod's own reference worker
(runpod-workers/worker-basic), whose layout is `rp_handler.py` at the repo root
with `runpod.serverless.start({"handler": handler})` as the last line.

Why this file exists at all, rather than starting from handler.py directly:
RunPod's GitHub integration statically scans the repo for that call before it
builds anything, and warns "runpod.serverless.start() handler not found in your
repo" when it cannot find it. handler.py is ~33 KB and the call sits near the
bottom, which is easy for a bounded prefix scan to miss. Keeping the entrypoint
here -- a dozen lines, canonical filename, canonical shape -- removes any
ambiguity for that scanner without touching how the worker actually runs.

The real implementation lives in handler.py. Importing it is side-effect free;
`boot()` is what downloads weights, stages them and constructs the pipeline.
"""

import runpod

from handler import boot, handler

if __name__ == "__main__":
    boot()
    runpod.serverless.start({"handler": handler})
