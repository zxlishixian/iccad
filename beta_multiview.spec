# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["/home/lishixian/iccad/beta_multiview_inference.py"],
    pathex=["/home/lishixian/iccad"],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "triton", "transformers", "datasets",
        "pyarrow", "matplotlib", "pandas", "IPython", "pytest", "tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="regr_fail_bucketing_multiview",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="regr_fail_bucketing_multiview",
)
