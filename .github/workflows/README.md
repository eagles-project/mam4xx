# MAM4xx Automated Testing

This document contains a brief description of how autotesting is conducted for MAM4xx.
Additionally, any helpful notes and TODOs may be kept here to assist developers.

## Overview

We use [GitHub Actions](https://docs.github.com/en/actions) to drive our testing.[^gh-actions-ref]
To do this, testing is initialized via the top-level workflow, `MAM4xx Autotester`, which is triggered by either a pull request (PR), events related to that PR, nightly, or manually from the repository's [**Actions**](https://github.com/eagles-project/mam4xx/actions) page (see the section on [triggers](#triggering-the-testing-workflow), below, for more details).

### Test Hardware and Compiler Configurations

#### GPU-based Testing

| Test Name                         | GPU Brand | GPU Type | Micoarchitecture | Compute Capability | Machine | Compilers                    |
| --------------------------------- | --------- | -------- | ---------------- | ------------------ | ------- | ---------------------------- |
| GPU AT2 gcc 12.3 cuda 12.1        | NVIDIA    | H100     | Hopper           | 9.0                | blake   | `gcc` 12.3.0/`nvcc` 12.1.105 |

#### CPU-based Testing

**Note:** These are the current specs for GitHub's Ubuntu 22.04 runner and are subject to change.

| Test Name                                    | OS                   | Machine        | Compiler   |
| -------------------------------------------- | -------------------- | -------------- | ---------- |
| GitHub CPU Auto-test Ubuntu 22.04[^gh-ubu2204] | Linux - Ubuntu 22.04 | GitHub Runners | `gcc` 12.3 |

### The Flow of the CI Workflow

Upon a trigger occurring, the top-level `MAM4xx Autotester` workflow is called.
This workflow handles concurrency, tagging jobs according to the combination of

- ***Workflow***
  - E.g., `GitHub CPU Auto-test Ubuntu 22.04` vs. `GPU AT2 gcc 12.3 cuda 12.1`
- ***GitHub Ref***
  - In the case of a PR trigger, this is essentially the number of the PR.
  - When running manually, this is the name of the branch being tested.
- ***Architecture***
  - E.g., GPU vs. CPU

Based on the trigger and/or inputs, `MAM4xx Autotester` dispatches sub-workflows, of which the possible choices are:

#### GPU AT2 `gcc` 12.3 `cuda` 12.1

- The full version of this test runs a "matrix-strategy" test running all combinations of
    - **Precision:** `[single, double]`
    - **Build Type:** `[Debug, Release]`
- The unit/validation tests that are run are determined by the MAM4xx CMake/CTest configuration.
- ***Note:*** AT2 = "Autotester 2," the second generation of a Sandia-developed GitHub-based testing product.
- See the [AT2 README](./AT2-README.md) for details about the implementation of the AT2 product.

#### GitHub CPU Auto-test Ubuntu 22.04

- The full version of this test runs a "matrix-strategy" test running all combinations of
    - **Precision:** `[single, double]`
    - **Build Type:** `[Debug, Release]`
- The unit/validation tests that are run are determined by the MAM4xx CMake/CTest configuration.
- The `[double, Debug]` test configuration also includes a code coverage check, followed by uploading the report to [codecov.io](https://app.codecov.io/gh/eagles-project/mam4xx).
    - If triggered by a PR, a comment is added to the PR Conversation that summarizes the Codecov report.
    - See this [PR comment](https://github.com/eagles-project/mam4xx/pull/437#issuecomment-2842974905) for an example.

#### `clang-format` Check

- Runs `clang-format` (v14) to verify whether all MAM4xx code in `src/` is in compliance with the style guidelines provided by the `.clang-format` file in the projects root directory.
- Currently, the style specification is merely "based on the [LLVM style](https://llvm.org/docs/CodingStandards.html)."

---

#### Triggering the Testing Workflow

This autotesting workflow is triggered by opening a pull request to `main` and
also by a handful of actions related to such a PR that is already open, including:

- `reopened`
- `ready_for_review`
  - I.e., converted to ***Ready for Review*** from ***Draft***
- `synchronize`
  - E.g., pushing a new commit or force pushing after rebase

The workflow may also be run manually by members of the `snl-testing` team--that is, via

> **Actions** -> **MAM4xx Autotester** -> **Run Workflow** -> *Choose Options from Dropdown Menu*

The current options when manually triggering a workflow are:

- Branch
- Test Machine Architecture
  - Current Options:
    - `GPU-NVIDIA_H100`
    - `CPU-Ubuntu_22-04`
    - `ALL`
- Floating-point Precision
  - Current Options:
    - `single`
    - `double`
    - `ALL`
- Build Type
  - Current Options:
    - `Debug`
    - `Release`
    - `ALL`

The other way to manually trigger a workflow is via

> **Actions** -> `<Previously-run SNL-AT2 Workflow/Job>` -> **Re-run `[all,this]` job(s)**.

##### Notes on Triggering AT2 Jobs

###### `tl;dr`

SNL prohibits individuals from running code on their machines unless the individual running the code has a valid user account on the machine. This restricts whether AT2-based testing is automatically triggered by a PR or whether a user can manually trigger AT2 testing.

###### Details

To satisfy the above restrictions, there are 2 GitHub Teams that are a part of the `eagles-project` GitHub Project.

- `snl-testing`
  - These are developers that have valid SNL user accounts on the target machines.
- `snl-testing-admins`
  - This is a subset of the `snl-testing` team that have permission to trigger AT2-based autotesting on PRs that modify the the autotesting behavior.
  - That is to say, PRs that modify files in the `.github` directory

Refer to the section on [Other Types of Job Control](./AT2-README.md#other-types-of-job-control) in the AT2 README for more details.

### Tokens

- Uploading the [code coverage report](#github-cpu-auto-test-ubuntu-2204) to Codecov requires a token that is stored in the repository as an Actions Secret.
  - The secrets variable is named `CODECOV_TOKEN`, and it appears that the token does not expire.

## TODO

- [x] Update job control section of README after the behavior is made clear.
  - @mjschmdt271
- [ ] Include a script to generate plots from within testing container?
  - @jaelynlitz?
- [x] Unify all CI into a single top-level yaml file that calls the sub-cases.
  - This should provide finer control over what runs and when.
  - @mjschmidt271
- [ ] Add testing for AMD GPUs on `caraway`.
  - @jaelynlitz - WIP

### Low-priority

- [ ] Add CPU testing on `mappy` because "heck, why not?"

[^gh-actions-ref]: While GitHub Actions can be a bit tricky at first, the docs are pretty decent and have lots of examples. Of particular use are the [**Quickstart**](https://docs.github.com/en/actions/writing-workflows/quickstart) and the **Write Workflows** section and its subsections titled ***Choose {when,where,what} workflows…*** Finally, an internet search typically reveals a handful of answers from stackoverflow and others.
[^gh-ubu2204]: Current specs for [hardware](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories) and [software](https://github.com/actions/runner-images/blob/main/images/ubuntu/Ubuntu2204-Readme.md) according to GitHub.
