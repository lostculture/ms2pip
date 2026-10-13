# MS2RescoreAdapter crash: `hdf5plugin` has no attribute `Blosc`

**Date:** 2026-10-13
**Results file:** `../data/20261012/F002460.msr`
**Report URL:** `http://localhost/mascot/cgi/master_results_2.pl?file=..%2Fdata%2F20261012%2FF002460.msr;_sigthreshold=0.01371;_target_fdr=1`
**Action:** Refining with machine learning, MS2Rescore with `deeplc_model=auto`

## Summary

`MS2RescoreAdapter.exe` stops while it is still importing its Python libraries, before any ms2pip or DeepLC code runs. The failure is inside the adapter's bundled dependencies (`psims` → `hdf5plugin`). **It is not an ms2pip problem.** Choosing a different DeepLC model wouldn't help, because the crash happens before the model setting is read.

## Error chain

```
insert_predicted_data.pl failed with exit code 1
  └─ MS2RescoreAdapter.exe ... deeplc_model=auto  → exit code 1
       MS2RescoreAdapter.py, line 189
       matrix_science\monkey_patching.py, line 57, in change_global_state
         import psims
           psims\__init__.py, line 12
           psims\mzmlb\__init__.py, line 2
           psims\mzmlb\writer.py, line 64
       AttributeError: module 'hdf5plugin' has no attribute 'Blosc'
       [PYI-89740:ERROR] Failed to execute script 'MS2RescoreAdapter'
```

Everything up to the adapter launch worked: the PIN file was read (58 columns, 73,444 PSMs across 8 chunks) and the `.psm` query lists were written.

## Analysis

- The adapter's `change_global_state` imports `psims` (MS2Rescore's mzML/mzMLb I/O library). At line 64, `psims/mzmlb/writer.py` refers to `hdf5plugin.Blosc`.
- psims handles a *missing* `hdf5plugin` gracefully: it falls back and warns "hdf5plugin is missing". ms2pip hides that warning at `ms2pip/__init__.py:19`, and that filter is the only place ms2pip mentions either package. In this crash, though, the import **succeeded** but returned a module without `Blosc`, and psims doesn't handle that case.
- The `pyimod02_importers.py` frames show that the exe was built with PyInstaller.
- Current `hdf5plugin` releases (up to 6.x/7.x) still have `Blosc`, so a version that removed it is unlikely. More likely causes, roughly in order:
  1. **Incomplete PyInstaller bundle.** PyInstaller packed `hdf5plugin`'s `__init__` but missed its submodules or compiled filter files, so the module loads without its filter classes.
  2. **Wrong module loaded.** The exe picked up a stub or very old `hdf5plugin`, or a different module of the same name.
  3. **Version mismatch** between the frozen `psims` and `hdf5plugin` (less likely, given the point above).

None of these could be confirmed: `psims` and `hdf5plugin` aren't installed in the dev environment where this was investigated, and the adapter's build files weren't checked.

## Status / possibly already fixed

The Mascot/adapter install is being left as it is for now because searches are running. A newer adapter or Mascot release may already fix this, so check release notes before rebuilding anything.

## Workaround while searches run

The error page offers **"load database search results (excluding refining with machine learning)"**. Use that to view F002460 without rescoring until the adapter is fixed.

## Suggested fix (later, at the adapter build)

1. In the Python environment the exe was built from:
   ```
   pip show hdf5plugin psims
   python -c "import hdf5plugin; print(hdf5plugin.version, hdf5plugin.__file__, hasattr(hdf5plugin, 'Blosc'))"
   ```
   - `False` → version/package problem: reinstall or pin `hdf5plugin`, or upgrade `psims`.
   - `True` → bundling problem: add `collect_all('hdf5plugin')` (or matching `hiddenimports` and `binaries`) to the PyInstaller `.spec` file and rebuild.
2. Fallback: if the adapter never writes mzMLb, leave `hdf5plugin` out of the bundle completely. psims will then take its "hdf5plugin missing" path and continue.
3. Report upstream (Matrix Science adapter, and possibly psims) if a current release still has the problem.

## References

- hdf5plugin docs: https://hdf5plugin.readthedocs.io/
- hdf5plugin API index (`Blosc` still present): https://hdf5plugin.readthedocs.io/en/stable/genindex.html
- hdf5plugin release listing: https://simple-repository.app.cern.ch/project/hdf5plugin
