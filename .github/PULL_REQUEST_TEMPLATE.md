<!--
IMPORTANT NOTES

1. Use GH flavored markdown when writing your description:
   https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax

2. If all boxes in the PR Checklist cannot be checked, this PR should be marked as a draft.

3. DO NOT DELTE ANYTHING FROM THIS TEMPLATE. If a section does not apply to you, simply write
   "N/A" in the description.

4. Code snippets to highlight new, modified, or problematic functionality are highly encouraged,
   though not required. Be sure to use proper code higlighting as demonstrated below.

   ```python
    def a_func():
        return 1

    a = 1
    b = a_func()
    print(a + b)
    ```
-->

<!--The title should clearly define your contribution succinctly.-->
# Add meaningful title here

<!-- Describe your feature here. Please include any code snippets or examples in this section. -->


## PR Checklist

<!--Tick these boxes if they are complete, or format them as "[x]" for the markdown to render. -->
- [ ] `RELEASE.md` has been updated to describe the changes made in this PR
- [ ] Documentation
  - [ ] Docstrings are up-to-date
  - [ ] Related `docs/` files are up-to-date, or added when necessary
  - [ ] Documentation has been rebuilt successfully
  - [ ] Examples have been updated
- [ ] Tests pass (If not, and this is expected, please elaborate in the tests section)
- [ ] PR description thoroughly describes the new feature, bug fix, etc.

## Related issues

<!--If one exists, link to a related GitHub Issue.-->


## Impacted areas of the software

<!--
Replace the below example with any added or modified files, and briefly describe what has been changed or added, and why.
-->
- `path/to/file.extension`
  - `method1`: What and why something was changed in one sentence or less.

## Additional supporting information

<!--Add any other context about the problem here.-->


## Test results, if applicable

<!--
Add the results from unit tests and regression tests here along with justification for any
failing test cases.
-->


<!--
__ For NREL use __
Release checklist:
- [ ] Update the version in hopp/__init__.py
- [ ] Verify docs builds correctly
- [ ] Create a tag on the main branch in the NREL/HOPP repository and push
- [ ] Ensure the Test PyPI build is successful
- [ ] Create a release on the main branch
-->
