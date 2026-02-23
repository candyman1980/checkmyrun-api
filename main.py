# add near the other routes in main.py
import sys
import importlib

@app.get("/_diag_openai")
def diag_openai():
    info = {"python_version": sys.version, "OPENAI_API_KEY_set": bool(OPENAI_API_KEY)}
    try:
        # real installed package version (if importable)
        import openai as _openai_pkg
        info["openai_installed_version"] = getattr(_openai_pkg, "__version__", "unknown")
    except Exception as e:
        info["openai_import_error"] = str(e)
        return info

    # Inspect the SDK class you've been using
    try:
        # try to construct the client object if possible (do NOT call remote)
        ClientClass = getattr(_openai_pkg, "OpenAI", None)
        info["client_class_present"] = ClientClass is not None
        if ClientClass:
            client_dir = dir(ClientClass())
            info["client_dir_sample"] = [n for n in client_dir if n in ("responses", "chat", "completions")][:10]
            info["has_responses"] = "responses" in client_dir
        else:
            info["client_dir_sample"] = []
            info["has_responses"] = False
    except Exception as e:
        info["client_diag_error"] = str(e)
        info["has_responses"] = False

    # also show any local files named openai that might shadow the package
    try:
        import os
        repo_files = []
        for root, dirs, files in os.walk(".", topdown=True):
            for f in files:
                if f.lower().startswith("openai"):
                    repo_files.append(os.path.join(root, f))
            # don't walk node_modules / .venv etc to keep this quick
            if ".venv" in dirs:
                dirs.remove(".venv")
        info["local_openai_files"] = repo_files[:10]
    except Exception:
        info["local_openai_files"] = []

    return info
