# GitHub Blame Audit Task

## Instruction
For each repo and code_string in the CSV:

1. Find the file in the repo that contains the code_string.
2. Navigate to the BLAME VIEW for that file. Find the line(s) matching the code_string.
3. Check how long ago the last change to that code was.
4. If the code was modified within the past year:
   - Go into the commit and read the diff.
   - Determine if it may have materially changed how something was calculated.
   - If so, flag it and take screenshots to document.
5. If the code was NOT modified in the past year:
   - Check the code BELOW the function — look for code that changes how the function operates that WAS modified within the past year.
   - If found, determine if this could have materially affected the outcome of the function/calculation.
   - Document with screenshots if so.

## Input CSV
input.csv — 5 code strings across numpy, pandas, and scikit-learn repos.

## Samples
| repo | code_string |
|---|---|
| numpy/numpy | `def linspace(start stop num` |
| numpy/numpy | `def percentile(a q axis` |
| numpy/numpy | `def corrcoef(x y rowvar` |
| pandas-dev/pandas | `def _cython_agg_general` |
| scikit-learn/scikit-learn | `def _compute_log_likelihood` |

## Results
5/5 completed. Pioneer-follower pattern with code analysis tools (get_recent_changes, get_commit_diff, get_function_source).

- **linspace**: Not modified within year. 3 nearby commits checked — all cosmetic (f-string, pylint, docstring). No material change.
- **corrcoef**: Modified Oct 2025. Removed deprecated `bias`/`ddof` params. Material API change.
- **_cython_agg_general**: Modified Oct 2025. Added `numeric_only` validation raising ValueError. Material behavior change.
- **percentile**: Def line from 2021, but nearby `interpolation` param removed Oct 2025. Material change.
- **_compute_log_likelihood**: Function no longer exists on main (removed ~2014). Correctly identified.
