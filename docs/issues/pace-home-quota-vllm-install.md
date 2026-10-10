# Installing vLLM filled the PACE home-directory quota

**Found:** 2026-10-09, while setting up L7. **Severity:** operational (fixed within minutes).

`conda create -n vllm` plus `pip install vllm` wrote the env (6.5 GB) and pip's cache
(6.9 GB) into `$HOME`, which has a 20 GB quota on PACE. The install failed with
`OSError: [Errno 122] Disk quota exceeded`, and a full home directory can break other
jobs that write logs there. Both directories were deleted at once (home back to 7 GB of
20 GB).

**Fix:** an existing vLLM 0.31.0 env already lives in project storage
(`~/ps-simpliearn-0/envs/vllm`). The L7 job activates it read-only and finds PlayParse
via `PYTHONPATH` instead of `pip install`-ing into it. Lesson: on PACE, conda envs and
pip caches belong in project storage (1 TB), never `$HOME`.
