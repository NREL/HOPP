# Contributing

We welcome contributions in the form of bug reports, bug fixes, improvements to the documentation,
ideas for enhancements, or the enhancements themselves!

You can find a [list of current issues](https://github.com/NREL/HOPP/issues) in the project's
GitHub repo. Feel free to tackle any existing bugs or enhancement ideas by submitting a
[pull request](https://github.com/NREL/HOPP/pulls).

## Bug Reports

* Please include a short (but detailed) Python snippet or explanation for reproducing the problem
  Attach or include a link to any input files that will be needed to reproduce the error.
* Explain the behavior you expected, and how what you got differed.

## Pull Requests

* Please reference relevant GitHub issues in your commit message using `GH123` or `#123`.
* Changes should be [PEP8](http://www.python.org/dev/peps/pep-0008/) compatible.
* Keep style fixes to a separate commit to make your pull request more readable.
* Docstrings are required and should follow the
  [Google style](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).
* When you start working on a pull request, start by creating a new branch pointing at the latest
  commit on [main](https://github.com/NREL/HOPP).
* The HOPP copyright policy is detailed in the
  [`LICENSE`](https://github.com/NREL/HOPP/blob/main/LICENSE).

## Documentation

Whenever code is modified or added, the documentation should be updated in the following ways:

1. The docstrings should represent the current state of the modified classes, methods, etc. using
   [Google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).
   If the docstrings were out of date from when your contribution was started, then please update
   them to the best of your ability.
2. Add or update the documentation site configuration files in `docs/`. If no page existed or if it
   was out of date, please take the time to add a new page/section/chapter as you see fit.

### Build the docs

If you haven't already, please install the developer tools using the "develop" flag. For instance,
if you use an editable installation: `pip install -e ".[develop]"`.

```bash
jupyter-book build docs/
```

Then open the docs in your browser:

```bash
open docs/_build/html/index.html
```

Be sure to fix any build errors or warnings related to the formatting. Please also check that the
updated page(s) are built as expected by inspecting the locally-built site.

## Tests

The test suite can be run using `pytest tests/hopp`. Individual test files can be run by specifying
them:

```bash
pytest tests/hopp/test_hybrid.py
```

and individual tests can be run within those files

```bash
pytest tests/hopp/test_hybrid.py::test_hybrid_wind_only
```

When you push to your fork, or open a PR, your tests will be run against the
[Continuous Integration (CI)](https://github.com/NREL/HOPP/actions) suite. This will start a build
that runs all tests on your branch against multiple Python versions, and will also test
documentation builds.
