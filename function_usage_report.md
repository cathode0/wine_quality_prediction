# Function Usage Report

Checked this repository for direct uses of the following Python functions:

- `str()`
- `int()`
- `float()`
- `complex()`
- `input()`
- `output()`

## Result

No direct uses of these functions were found in the source files.

## Search Patterns Used

```powershell
rg -n --glob '!__pycache__/**' --glob '!*.pyc' "\b(str|int|float|complex)\s*\("
rg -n --glob '!__pycache__/**' --glob '!*.pyc' "\b(input|output)\s*\("
```
