#!/usr/bin/env python
"""Upload to Hugging Face while patching mirror DNS in-process."""

from __future__ import annotations

import argparse
import importlib
import os
import socket
import sys
import time
from pathlib import Path


def patch_dns(host: str, ip: str) -> None:
    original_getaddrinfo = socket.getaddrinfo

    def patched_getaddrinfo(query_host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        if query_host == host:
            query_host = ip
        return original_getaddrinfo(query_host, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo


def patch_hf_mirror_org_urls(replacement_endpoint: str) -> None:
    """Rewrite hub-generated hf-mirror.org LFS completion URLs to the working endpoint."""
    import huggingface_hub.lfs as hub_lfs

    replacement_endpoint = replacement_endpoint.rstrip("/")
    original_http_backoff = hub_lfs.http_backoff

    def rewrite_url(url):
        if isinstance(url, str):
            if url.startswith("https://hf-mirror.org"):
                return replacement_endpoint + url[len("https://hf-mirror.org") :]
            if url.startswith("http://hf-mirror.org"):
                return replacement_endpoint + url[len("http://hf-mirror.org") :]
        return url

    def patched_http_backoff(method, url=None, *args, **kwargs):
        if url is None and "url" in kwargs:
            kwargs["url"] = rewrite_url(kwargs["url"])
            return original_http_backoff(method, *args, **kwargs)
        return original_http_backoff(method, rewrite_url(url), *args, **kwargs)

    hub_lfs.http_backoff = patched_http_backoff


def clear_partial_imports() -> None:
    prefixes = ("huggingface_hub", "httpcore", "httpx", "yaml")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def import_hub_api(max_attempts: int = 8):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            from huggingface_hub import HfApi
            from huggingface_hub._commit_api import CommitOperationAdd

            return HfApi, CommitOperationAdd
        except (ImportError, ModuleNotFoundError) as error:
            last_error = error
            clear_partial_imports()
            wait_s = min(2 ** (attempt - 1), 10)
            print(f"Import huggingface_hub failed on attempt {attempt}/{max_attempts}: {error}")
            if attempt < max_attempts:
                print(f"Retrying import in {wait_s}s...")
                time.sleep(wait_s)
    raise RuntimeError(f"Failed to import huggingface_hub after {max_attempts} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--mirror-org-ip", default="160.16.86.14")
    parser.add_argument("--commit-message", default="Upload PI0.5 fold towel checkpoint")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    patch_dns("hf-mirror.org", args.mirror_org_ip)

    HfApi, CommitOperationAdd = import_hub_api()

    patch_hf_mirror_org_urls(args.endpoint)

    api = HfApi(endpoint=args.endpoint)
    local_dir = args.local_dir.resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)

    path_prefix = args.path_in_repo.strip("/")
    files = sorted(path for path in local_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found under {local_dir}")

    operations = []
    for path in files:
        relative_path = path.relative_to(local_dir).as_posix()
        path_in_repo = f"{path_prefix}/{relative_path}" if path_prefix else relative_path
        operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=path))

    print(f"Uploading {args.local_dir} -> {args.repo_id}/{args.path_in_repo}")
    print(f"endpoint={args.endpoint}, patched hf-mirror.org -> {args.mirror_org_ip}")
    print(f"rewriting hf-mirror.org LFS completion URLs -> {args.endpoint}")
    print(f"files={len(operations)}, upload_threads=1")
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        operations=operations,
        commit_message=args.commit_message,
        num_threads=1,
    )
    print(result)


if __name__ == "__main__":
    main()
