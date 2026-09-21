# Releasing typedecide

A release is a git tag. Pushing `vX.Y.Z` runs
[`.github/workflows/release.yml`](../../.github/workflows/release.yml), which builds,
checks, publishes and cuts the GitHub Release. Nobody uploads from a laptop, and there
is no PyPI API token anywhere: the workflow authenticates with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

| Tag | Goes to | GitHub environment | GitHub Release |
|---|---|---|---|
| `vX.Y.ZrcN` (for example `v0.2.0rc1`) | TestPyPI | `testpypi` | pre-release |
| `vX.Y.Z` (for example `v0.2.0`) | PyPI | `pypi` | latest |

Two facts shape everything below. **A PyPI upload cannot be undone**, and **a version
number can never be reused**, not even after deleting the release. So every check runs
before the upload, and the fix for a bad release is always a new version.

The version lives in exactly one place: `__version__` in
[`src/typedecide/__init__.py`](src/typedecide/__init__.py). `pyproject.toml` reads it
from there (`[tool.hatch.version]`), and the workflow refuses to run if the tag
disagrees with it.

## Pre-flight checklist

Run from `packages/typedecide` on an up-to-date `main`.

- [ ] CI (`checks`) is green on the commit you are about to tag.
- [ ] Working tree is clean: `git status` shows nothing.
- [ ] Tests, lint, types and the coverage floor pass locally:

  ```bash
  uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]" build twine
  .venv/bin/python -m pytest --cov
  .venv/bin/python -m ruff check src tests
  .venv/bin/python -m mypy --strict src/typedecide
  ```

- [ ] The package builds and the metadata renders:

  ```bash
  rm -rf dist && .venv/bin/python -m build && .venv/bin/python -m twine check --strict dist/*
  ```

- [ ] `README.md` reads correctly on its own. PyPI renders it as the project page and
      does not resolve relative links, so links to files in this repository must be
      absolute `https://github.com/...` URLs.
- [ ] `CHANGELOG.md` has everything user-visible under `[Unreleased]`.
- [ ] Any new runtime dependency is in `dependencies` or the right extra, and
      `import typedecide` still imports neither torch nor transformers.
- [ ] For anything other than a patch release, a release candidate has been through
      TestPyPI first (see below).

## Cutting a release

1. **Pick the version.** [SemVer](https://semver.org): patch for fixes, minor for
   additions, major for breaking changes. While the version is `0.y.z`, a breaking
   change bumps the minor. Versions are [PEP 440](https://peps.python.org/pep-0440/):
   `0.2.0` or `0.2.0rc1`. No `v` inside the version, no hyphen before `rc`.

2. **Bump the version.** Edit the one line in `src/typedecide/__init__.py`:

   ```python
   __version__ = "0.2.0"
   ```

3. **Update the changelog.** In `CHANGELOG.md`, rename `## [Unreleased]` to
   `## [0.2.0] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and update
   the link references at the bottom:

   ```markdown
   [Unreleased]: https://github.com/ardada2468/typedecide/compare/v0.2.0...HEAD
   [0.2.0]: https://github.com/ardada2468/typedecide/compare/v0.1.0...v0.2.0
   ```

   The heading format matters. The workflow extracts the section whose heading starts
   `## [0.2.0]` and uses it as the GitHub Release body; with no such section, or an
   empty one, the release fails before anything is uploaded. A release candidate does
   not need its own section: `0.2.0rc1` falls back to `[0.2.0]`, then `[Unreleased]`.

4. **Commit and merge** through the normal pull-request path:

   ```bash
   git commit -am "Release 0.2.0"
   ```

5. **Tag the merged commit on `main` and push the tag.** Use an annotated tag.

   ```bash
   git checkout main && git pull
   git tag -a v0.2.0 -m "typedecide 0.2.0"
   git push origin v0.2.0
   ```

6. **Approve the deployment** if the `pypi` environment has required reviewers (it
   should): Actions > release > the run > *Review deployments*.

7. **Verify** (see below).

### Release candidates

Identical, with an `rc` version: set `__version__ = "0.2.0rc1"`, merge, then
`git tag -a v0.2.0rc1 -m "typedecide 0.2.0rc1" && git push origin v0.2.0rc1`. When the
candidate is good, bump to `0.2.0` and go through the steps above. Release candidates
are published to TestPyPI only and never reach PyPI.

## What the workflow does

`release.yml` runs on any pushed tag matching `v[0-9]+.[0-9]+.[0-9]+*`. The workflow
holds `contents: read` and nothing else; each job adds only what it needs.

1. **`build`** (no extra permissions)
   - Checks out the tag and reads `__version__` from `src/typedecide/__init__.py`
     without importing it. **Fails if the tag is not exactly `v` + `__version__`**, or
     if the version is not `X.Y.Z` or `X.Y.ZrcN`.
   - Extracts the matching `CHANGELOG.md` section into the release notes, and fails if
     there is none.
   - `python -m build` in `packages/typedecide`: an sdist, and a wheel built *from that
     sdist*, so a broken sdist cannot hide behind a good wheel.
   - `twine check --strict dist/*`: metadata is valid and the README renders.
   - Asserts the wheel contains `py.typed`, all four subpackages (`data`, `training`,
     `evaluation`, `export`), the CLI, the licence, and nothing outside the package.
   - Installs the wheel into a clean virtualenv and, from outside the checkout, runs
     `python -c "import typedecide; print(typedecide.__version__)"` (and checks it
     equals the tag), `typedecide --help`, and an assertion that `import typedecide`
     put neither `torch` nor `transformers` in `sys.modules`.
   - Uploads `dist/` and the release notes as workflow artefacts. The publish jobs
     upload these exact files; nothing is rebuilt after it has been tested.
2. **`publish-testpypi`**: only when the tag contains `rc`. Environment `testpypi`,
   `permissions: id-token: write`, `pypa/gh-action-pypi-publish` pointed at
   `https://test.pypi.org/legacy/`.
3. **`publish-pypi`**: only when the tag does not contain `rc`. Environment `pypi`,
   `permissions: id-token: write`, `pypa/gh-action-pypi-publish`.
4. **`github-release`**: after whichever publish job ran has succeeded.
   `permissions: contents: write`. Creates the GitHub Release for the tag with the
   sdist and wheel attached and the changelog section as the body; `rc` tags are marked
   as pre-releases.

The publish action also generates and uploads [PEP 740](https://peps.python.org/pep-0740/)
attestations, so each file on PyPI is verifiably tied to this repository and workflow.

If a release fails in `build`, nothing has been published: delete the tag
(`git push --delete origin v0.2.0 && git tag -d v0.2.0`), fix, and tag again. If it
fails *after* a publish job succeeded, the version is spent; re-run the failed job, do
not re-tag.

## Verifying a release

### On TestPyPI (release candidates)

TestPyPI does not mirror PyPI, so the dependencies (`numpy`, `pyyaml`) must come from
the real index:

```bash
uv venv /tmp/td-verify && source /tmp/td-verify/bin/activate
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  "typedecide==0.2.0rc1"
# with plain pip, drop the --index-strategy line

cd /tmp   # not the repo, or Python imports src/ instead of the installed wheel
python -c "import typedecide; print(typedecide.__version__)"   # 0.2.0rc1
typedecide --help
python -c "import sys, typedecide; assert 'torch' not in sys.modules"
```

Then look at <https://test.pypi.org/project/typedecide/>: the README renders, the
sidebar links work, the licence reads MIT, the classifiers are right, and both a
`.tar.gz` and a `.whl` are listed.

### On PyPI (final releases)

```bash
uv venv /tmp/td-verify && source /tmp/td-verify/bin/activate
uv pip install "typedecide==0.2.0"
cd /tmp && python -c "import typedecide; print(typedecide.__version__)" && typedecide --help
```

- <https://pypi.org/project/typedecide/> shows the new version and a rendered README.
- The file listing shows a provenance/attestation entry for each file.
- <https://github.com/ardada2468/typedecide/releases> has the release, with the
  changelog section as its body and both files attached.
- A type-checker sees the package as typed: `mypy -c "import typedecide"` reports no
  missing-stubs error.

## Yanking a bad release

Yank, do not delete. A yanked release stays downloadable for anyone who pinned it
exactly (`typedecide==0.2.0`), so existing lockfiles keep working, but resolvers stop
selecting it for `pip install typedecide` or `typedecide>=0.1`. Deleting breaks pinned
users and still does not free the version number.

1. <https://pypi.org/manage/project/typedecide/releases/>
2. On the bad version: **Options > Yank**.
3. Give a reason; pip shows it to anyone who installs the yanked version. For example:
   `0.2.0 mis-scores decisions under --debias calibrated; use 0.2.1`.
4. Ship the fix as the next patch version (`0.2.1`) through the normal process, and
   record it under **Fixed** in the changelog, naming the yanked version.
5. Edit the GitHub Release for the bad tag: prefix the title with `[YANKED]`, say why,
   and untick *Set as the latest release*. Leave the git tag in place; it is history.

Yanking is reversible (**Options > Un-yank**). Delete a release only if it contains
something that must not be public, such as a committed secret, and in that case rotate
the secret first: assume everything uploaded has already been copied.

## One-time setup

Done once per index by the account that will own the project. Until this is done the
publish jobs fail with `invalid-publisher`.

### 1. PyPI: register a pending Trusted Publisher

The project does not exist on PyPI yet, so this is a *pending* publisher: it creates
the project on the first successful upload.

1. Sign in at <https://pypi.org>. Two-factor authentication must be enabled on the
   account (Account settings > Two factor authentication).
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **Add a new pending publisher**, choose the **GitHub** tab and enter exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `typedecide` |
   | Owner | `ardada2468` |
   | Repository name | `typedecide` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

The workflow name is the file name only, not the path. Every field is matched exactly
against the OIDC token GitHub issues, so a different repository name, a renamed workflow
file or a different environment name all fail as `invalid-publisher`. If the GitHub
repository is not called `typedecide`, enter its real name here and correct the URLs in
`pyproject.toml` and `CHANGELOG.md`.

A pending publisher does **not** reserve the name. Until the first upload, anyone else
can still register `typedecide`, so publish soon after registering.

### 2. TestPyPI: the same, with a different environment

TestPyPI is a separate site with separate accounts. Sign in (or register, with 2FA) at
<https://test.pypi.org>, go to <https://test.pypi.org/manage/account/publishing/>, and
add a pending **GitHub** publisher with:

| Field | Value |
|---|---|
| PyPI Project Name | `typedecide` |
| Owner | `ardada2468` |
| Repository name | `typedecide` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` |

### 3. GitHub: create the two environments

Repository **Settings > Environments > New environment**, twice:

- **`pypi`**: enable **Required reviewers** and add yourself, so every real release
  waits for a human click. Under **Deployment branches and tags**, choose *Selected
  branches and tags* and add a **tag** rule `v*`, so only a release tag can use this
  environment.
- **`testpypi`**: same tag rule `v*`; reviewers optional.

Do **not** add any secrets to either environment. There is nothing to store.

### 4. GitHub: protect release tags (recommended)

**Settings > Rules > Rulesets > New tag ruleset**, target `v*`, restrict creation,
update and deletion to repository admins. A tag is what triggers a publish, so who can
push a tag is who can release.

### After the first release

The pending publisher becomes an ordinary Trusted Publisher, visible under
<https://pypi.org/manage/project/typedecide/settings/publishing/>. Never create an API
token for this project; if one exists, delete it.
