# Releasing a new version

A release is a git tag. Pushing `vX.Y.Z` triggers `.github/workflows/release.yml`,
which builds the sdist and wheel from `packages/typedecide`, runs `twine check`,
publishes, and creates the GitHub Release. Nobody uploads from a laptop. There is no
PyPI API token anywhere: the workflow authenticates with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

| Tag | Published to | GitHub environment | GitHub Release |
|---|---|---|---|
| `vX.Y.ZrcN`, for example `v0.2.0rc1` | TestPyPI | `testpypi` | pre-release |
| `vX.Y.Z`, for example `v0.2.0` | PyPI | `pypi` | latest |

The rule is literal: a tag containing `rc` goes to TestPyPI, and any other matching
tag goes to PyPI.

!!! danger "Two facts shape the whole procedure"
    **A PyPI upload cannot be undone, and a version number can never be reused**, not
    even after deleting the release. So every check runs before the upload, and the
    fix for a bad release is always a new version.

The version lives in one place: `__version__` in
`packages/typedecide/src/typedecide/__init__.py`. `pyproject.toml` reads it from there,
and the workflow refuses to run if the tag disagrees with it.

The maintainers' full procedure, including the one-time PyPI and GitHub setup, is
[`RELEASING.md`](https://github.com/ardada2468/typedecide/blob/main/packages/typedecide/RELEASING.md)
in the repository. This runbook is the operational summary.

## Symptoms

You are here because:

- A change is merged and needs to reach users, or
- A release run failed, or
- A published release is bad and must be withdrawn.

## Diagnosis

### Before tagging: is `main` releasable?

Run from `packages/typedecide` on an up-to-date `main` with a clean working tree.

```bash
python -m pip install -e ".[dev]" build twine
python -m pytest
python -m ruff check src tests
python -m mypy --strict src/typedecide
rm -rf dist && python -m build && python -m twine check --strict dist/*
```

- [ ] The `checks` workflow is green on the commit you will tag.
- [ ] `CHANGELOG.md` lists every user-visible change under `[Unreleased]`.
- [ ] `README.md` reads correctly on its own. PyPI renders it as the project page and
      does not resolve relative links.
- [ ] `import typedecide` still imports neither torch nor transformers.
- [ ] The documentation builds: `mkdocs build --strict`.
- [ ] For anything other than a patch release, a release candidate has been through
      TestPyPI first.

### After a failed run: which job failed?

| Failed job | What it means | Has anything been published? |
|---|---|---|
| `build`: *Tag must match `__version__`* | The tag is not exactly `v` + `__version__`, or the version is not `X.Y.Z` / `X.Y.ZrcN` | No |
| `build`: *Changelog must have a section* | No non-empty `## [X.Y.Z]` section in `CHANGELOG.md` | No |
| `build`: `twine check` | Package metadata is invalid or the README does not render | No |
| `build`: wheel contents or smoke test | A subpackage, `py.typed` or the licence is missing from the wheel, or the installed wheel does not import cleanly | No |
| `publish-pypi` / `publish-testpypi`: `invalid-publisher` | The Trusted Publisher registration does not match the repository, workflow file name or environment name | No |
| `publish-*`: file already exists | That version was already uploaded | It was, earlier |
| `github-release` | The package **is** published. Only the GitHub Release is missing | Yes |

## Fix

### Cut a release

1. **Pick the version.** [SemVer](https://semver.org): patch for fixes, minor for
   additions, major for breaking changes. While the version is `0.y.z`, a breaking
   change bumps the minor. Versions are PEP 440: `0.2.0` or `0.2.0rc1`. No `v` inside
   the version, and no hyphen before `rc`.

2. **Bump the version.** One line in `src/typedecide/__init__.py`:

    ```python
    __version__ = "0.2.0"
    ```

3. **Update the changelog.** Rename `## [Unreleased]` to `## [0.2.0] - YYYY-MM-DD`, add
   a fresh empty `## [Unreleased]` above it, and update the link references at the
   bottom of the file. The heading format matters: the workflow extracts the section
   whose heading starts `## [0.2.0]` and uses it as the GitHub Release body. A release
   candidate does not need its own section; `0.2.0rc1` falls back to `[0.2.0]`, then
   to `[Unreleased]`.

4. **Commit and merge** through the normal pull-request path.

5. **Tag the merged commit on `main` and push the tag.**

    ```bash
    git checkout main && git pull
    git tag -a v0.2.0 -m "typedecide 0.2.0"
    git push origin v0.2.0
    ```

6. **Approve the deployment** if the `pypi` environment has required reviewers:
   Actions, then the `release` run, then *Review deployments*.

For a release candidate, do the same with `__version__ = "0.2.0rc1"` and the tag
`v0.2.0rc1`. Release candidates are published to TestPyPI only.

### Recover from a failed run

- **Failed in `build`.** Nothing was published. Delete the tag, fix the problem, tag
  again:

    ```bash
    git push --delete origin v0.2.0 && git tag -d v0.2.0
    ```

- **Failed with `invalid-publisher`.** Correct the Trusted Publisher registration on
  PyPI (owner, repository name, workflow file name `release.yml`, environment name),
  then re-run the failed job. Do not re-tag.

- **Failed after a publish job succeeded.** The version is spent. Re-run the failed
  job. Do not re-tag and do not rebuild.

### Withdraw a bad release

Yank it. Do not delete it. A yanked release stays installable for anyone who pinned it
exactly, so existing lockfiles keep working, and resolvers stop selecting it for
unpinned installs. Deleting breaks pinned users and still does not free the version
number.

1. On PyPI: *Manage project*, *Releases*, the bad version, *Options*, *Yank*, with a
   reason. pip shows the reason to anyone installing that version.
2. Ship the fix as the next patch version through the normal process, and record it
   under **Fixed** in the changelog, naming the yanked version.
3. Edit the GitHub Release for the bad tag: prefix the title with `[YANKED]`, say why,
   and untick *Set as the latest release*. Leave the git tag in place.

Delete a release only if it contains something that must not be public, such as a
committed secret, and rotate the secret first.

## Verify

**A release candidate on TestPyPI.** TestPyPI does not mirror PyPI, so the
dependencies must come from the real index:

```bash
python -m venv /tmp/td-verify && . /tmp/td-verify/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ "typedecide==0.2.0rc1"
cd /tmp   # not the repo, or Python imports src/ and not the installed wheel
python -c "import typedecide; print(typedecide.__version__)"
typedecide --help
python -c "import sys, typedecide; assert 'torch' not in sys.modules"
```

**A final release on PyPI.**

```bash
python -m venv /tmp/td-verify && . /tmp/td-verify/bin/activate
pip install "typedecide==0.2.0"
cd /tmp && python -c "import typedecide; print(typedecide.__version__)" && typedecide --help
```

- The PyPI project page shows the new version with a rendered README, and both a
  `.tar.gz` and a `.whl` are listed.
- The GitHub Releases page has the release, with the changelog section as its body and
  both files attached.
- The documentation site has rebuilt from `main`
  ([`docs.yml`](https://github.com/ardada2468/typedecide/blob/main/.github/workflows/docs.yml)
  runs on every push to `main` that touches the package or `web/`).

## One-time setup

Until this is done, the publish jobs fail with `invalid-publisher`. It is done once per
index, by the account that will own the project.

| Where | What |
|---|---|
| PyPI, *Publishing*, *Add a new pending publisher* (GitHub tab) | Project `typedecide`, owner `ardada2468`, repository `typedecide`, workflow `release.yml`, environment `pypi` |
| TestPyPI, the same page | The same values with environment `testpypi` |
| GitHub, *Settings*, *Environments* | Create `pypi` with required reviewers and a tag rule `v*`. Create `testpypi` with the same tag rule |
| GitHub, *Settings*, *Rules* (recommended) | A tag ruleset on `v*` restricting who can create, update and delete release tags |

Every field of the publisher registration is matched exactly against the OIDC token
GitHub issues. The workflow name is the file name only, not its path. Do **not** add
secrets to either environment; there is nothing to store. If the GitHub repository is
not named `typedecide`, register its real name.
