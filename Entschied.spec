# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path.cwd()
llama_dir = project_root / "app" / "llama"
icon_path = project_root / "assets" / "entschied.ico"

llama_datas = []
if llama_dir.exists():
    for item in llama_dir.iterdir():
        if item.is_file():
            llama_datas.append((str(item), "llama"))

common_datas = [
    (str(project_root / "LICENSE"), "."),
]

block_cipher = None


a = Analysis(
    [str(project_root / "entschied" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=llama_datas + common_datas,
    hiddenimports=[
        "tinyjev",
        "tinyjev.agent",
        "tinyjev.families",
        "tinyjev.families.pointer",
        "tinyjev.backends",
        "tinyjev.backends.torch_backend",
        "tokenizers",
        "safetensors",
        "numpy",
        "httpx",
        "huggingface_hub",
        "entschied.jevk5",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torch_directml",
        "transformers",
        "mcp",
        "entschied.mcp",
        "entschied.gpu_backend",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Entschied",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Entschied",
)
