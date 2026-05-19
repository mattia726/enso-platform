# third_party/oner

Place the upstream Oner et al. open-source code and supplementary materials here.

Recommended layout:

```
third_party/oner/
  upstream/            # the original repo or tarball contents
  LICENSE              # copy the upstream license here
  NOTES.md             # any local notes about modifications
```

Do **not** modify upstream files in-place if you can avoid it. Prefer:

- keeping upstream code untouched under `upstream/`
- writing small adapter modules in our codebase that call into it
