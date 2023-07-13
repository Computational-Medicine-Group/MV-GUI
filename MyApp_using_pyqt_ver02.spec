# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['MyApp_using_pyqt_ver02.py'],
    pathex=[],
    binaries=[],
    datas=[(r'C:\Users\psk00\Downloads\Priyanka\MV-GUI\validation_template.docx', '.'),],  # Use raw string literal
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='Report Generator',
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
    icon=r'C:\Users\psk00\Downloads\Priyanka\MV-GUI\Insel_logo.ico',  # Use raw string literal
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Report Generator',
)
