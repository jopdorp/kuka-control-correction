# Tests for KUKA Vision Correction System

Minimal suite kept for now. Legacy tests against an older API were removed.

## Run

```bash
pytest -q
```

## Notes

- Only `pose_utils` unit tests are retained. They use the current API:
	- `pose_to_T(translation_mm: np.ndarray, rotation_abc_deg: np.ndarray)`
	- `invert_T(T: np.ndarray)`
	- `kuka_abc_to_rotation_matrix` / `rotation_matrix_to_kuka_abc`
- `conftest.py` adds `raspberry-pi/src` to `PYTHONPATH` for imports.
- ArUco and system tests will be reintroduced later with proper mocks.
