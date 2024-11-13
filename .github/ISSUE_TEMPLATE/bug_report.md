---
name: Bug report
about: Report a bug to help us improve
title: 'Bug report'
labels: "Type: Bug"
---

<!--
Thank you for taking the time to report a bug. If you aren't certain whether an issue
is a bug, please first open a Discussion. Before submitting, please reread your
description to ensure that other readers can reasonably understand the issue
you're facing and the impact on your workflow or results.

IMPORTANT NOTES

1. Replace all example text (contained in "<>") or anywhere specifically commenting to replace the
   text, leaving any guiding HTML comments in place (formatted like this large block so it won't
   show up in your Bug Report text.)
2. Use GH flavored markdown: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax,
   especially for code snippets, which should look like the following:
   ```python
   a = 1
   b = 2
   print(a + b)
   ```
3. Please be as thorough as possible when describing what went wrong, and what was expected from a
   correct solution. The amount of information required to describe the bug may differ, but more
   information is always helpful to ensure you receive the help you need.
-->

<!--The title should clearly define the issue succinctly.-->
# Add meaningful title here

<!-- Describe your bug/issue here using as much detail as necessary. -->


## How to reproduce

<!-- Describe how another person with no context can recreate this issue. -->


## Relevant output

<!-- Include any output, plots, tracebacks, or other means of communication here to add context to
the problem. All code and full tracebacks should be properly markdown formatted. -->


## System Information
<!-- Add your information here. -->
- OS: <macOS 12.4>
  <!-- e.g. Ubuntu 20.04 or macOS 10.12 -->
- Python version: <3.10.4>
  <!-- All OS: `python --version`-->
- HOPP version: <0.1.1>
  <!--
  Unix: pip freeze | grep hopp | awk -F"git@" '/git@/{print $2}' | awk -F"#egg" '/#egg/{print $1}'
  Windows: `pip list --format freeze | findstr hopp`
  -->
  - <Installed from source using an editable installation with developer tools: `pip install -e .[develop]`>
  - Commit hash: <commit-hash>
    <!--
    Unix: `pip freeze | grep hopp | awk -F"git@" '/git@/{print $2}' | awk -F"#egg" '/#egg/{print $1}'`
    Windows: `pip freeze | findstr hopp`, then copy the full git hash between "git@" and "#egg"
    -->

### Relevant library versions
<!--
Use `pip freeze` to gather the relevant versions, and use the markdown table formatting as
demonstrated below to replacing all relavant packages and their versions.
-->
  
  | Package | Version |
  | ------- | ------- |
  | numpy | <1.26.4> |
  | Pyomo | <6.8.0> |
  | scipy | <1.14.1> |
  | <another-relevant-package> | <version> |
