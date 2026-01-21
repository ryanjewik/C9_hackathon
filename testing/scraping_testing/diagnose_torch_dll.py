import os
import sys
import glob
import ctypes

print('Python executable:', sys.executable)
print('sys.prefix:', sys.prefix)
print('\nEnvironment PATH (end snippet):')
PATH = os.environ.get('PATH','')
for p in PATH.split(os.pathsep)[-10:]:
    print('  ', p)

# Try to locate torch package
try:
    import importlib.util
    spec = importlib.util.find_spec('torch')
    print('\nTorch spec:', spec)
except Exception as e:
    print('\nError finding torch spec:', e)

# Candidate torch lib dirs
candidates = []
site_packages = [os.path.join(sys.prefix, 'Lib', 'site-packages'), os.path.join(sys.prefix, 'lib', 'site-packages')]
for sp in site_packages:
    tdir = os.path.join(sp, 'torch', 'lib')
    if os.path.isdir(tdir):
        candidates.append(tdir)

print('\nCandidate torch lib directories:')
for c in candidates:
    print('  ', c)

# List DLLs in each candidate dir and try to LoadLibrary them to get Win32 error details
for c in candidates:
    dlls = glob.glob(os.path.join(c, '*.dll'))
    print(f'\nFound {len(dlls)} DLLs in {c}')
    for dll in dlls:
        print('\nTesting DLL:', dll)
        try:
            # Use LoadLibraryExW with LOAD_WITH_ALTERED_SEARCH_PATH (0x00000008) to mimic Windows loader behavior
            h = ctypes.windll.kernel32.LoadLibraryExW(dll, None, 0x00000008)
            if h:
                print('  Loaded OK -> handle:', h)
                ctypes.windll.kernel32.FreeLibrary(h)
            else:
                err = ctypes.windll.kernel32.GetLastError()
                # Format the error
                buf = ctypes.create_unicode_buffer(1024)
                ctypes.windll.kernel32.FormatMessageW(0x00001000, None, err, 0, buf, len(buf), None)
                print('  LoadLibraryExW failed with code', err, '->', buf.value)
        except Exception as e:
            print('  Exception while loading:', e)

# Try import torch to capture exception traceback (again) and print it
print('\nAttempting `import torch` to capture full traceback:')
try:
    import torch
    print('Imported torch OK:', torch.__version__)
except Exception as e:
    import traceback
    traceback.print_exc()

print('\nDiagnostic complete')
