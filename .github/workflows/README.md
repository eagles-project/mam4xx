
# Autotester2 (AT2) SNL Workflow for MAM4xx

This document contains a brief description of how AT2 is used to automate
testing on SNL hardware.
Additionally, any helpful notes and TODOs may be kept here to assist developers.

## Overview

AT2 is a Sandia-developed project for automating testing via GitHub Actions to
be run on self-hosted runners on the SNL network.
Part of what AT2 does is control access using information about the repository,
organization, user, etc. obtained via the GitHub API.
This is done for security/policy reasons and ensures that only those with
approved SNL computing accounts can run the CI code on SNL hardware.

### Test Hardware and Compiler Configurations

| Test Name | GPU Brand | GPU Type | Micoarchitecture | Compute Capability | Machine | Compilers |
|-|-|-|-|-|-|-|
| gcc_12-3-0_cuda_12-1 | NVIDIA | H100 | Hopper | 9.0 | blake | `gcc` 12.3.0/`nvcc` 12.1.105 |

### The Flow of the CI Workflow

AT2 runs on the target SNL machine and makes a handful of self-hosted runners
available to the MAM4xx repo.
This is all controlled by the **MAM4xx** SNL entity account that is linked to the
**mam4xxSNL** github account.
Each runner stays in a "holding pattern" until it is assigned a job via
GitHub Actions.
The holding pattern pulls the testing image from the AT2 Gitlab
repo (if necessary), runs the related container for 3 minutes, and then tears down and
starts over.
As of now, the image is of a UBI 8 system, with Spack-installed compilers and
all of the requisite TPLs to clone/build/run MAM4xx.

#### Triggering the Testing Workflow

This autotesting workflow is triggered by opening a pull request to `main` and
also by a handful of actions on such a PR that is already open, including:

- `reopened`
- `ready_for_review`
  - I.e., converted to ***Ready for Review*** from ***Draft***
- `synchronize`
  - E.g., pushing a new commit or force pushing after rebase

The workflow may also be run manually by members of the `snl-testing`
team--that is, via

> **Actions** -> **SNL-AT2 Workflow** -> **Run Workflow** -> *Choose Branch from Dropdown Menu*.

or

> **Actions** -> `<Previously-run SNL-AT2 Workflow/Job>` -> **Re-run `[all,this]` job(s)**.

The AT2 configuration on `blake` currently attempts to keep 3 runners available
to accept jobs at all times.
This workflow is configured to allow concurrent testing, so up to 3 test-matrix
configurations can run at once.
The concurrency setting is also configured to kill any active job if another
instance of this workflow is started for the same PR ref.

##### Other Types of Job Control

- If a PR contains changes to the `.github` directory, a member of the
  `snl-testing-admins` team must add the `CI-AT2_special_approval` tag to the
  PR in order to kick off the autotesting.
- For changes unrelated to the `.github` directory, any PR that is submitted
  by a member of the `snl-testing` team, and *only contains commits* from
  members of that team will automatically trigger this autotesting.
- In the case that the PR is submitted by someone who is not a member of the
  `snl-testing` team or contains commits from someone outside of that team,
  an approving review by someone on the `snl-testing` team is required to
  trigger autotesting.

###### Disclaimer

The above is according to Mike's current understanding of AT2 and may contain
minor inaccuracies.
This will be updated accordingly upon confirmation.

## Development Details

Most of the required configuration is provided by the AT2 docs and
instructional Confluence page (on the Sandia network :confused:--reach out if
you need access).
However, some non-obvious choices and configurations are listed here.

- To add some info to the testing output, we employ a custom action, cribbed
  from E3SM/EAMxx, that prints out the workflow's trigger.

### Hacks

- For whatever reason, Skywalker does not like building in the
  `gcc_12-3-0_cuda_12-1` container for the H100 GPU.
  - This appears to be an issue of the (Haero?) build not auto-detecting the
    correct Compute Capability (CC 9.0 => `sm_90`).
  - To overcome this, we first obtain the CC flag via `nvidia-smi` within the
    testing container.
  - Then, we employ `sed` to manually change the `default_arch="sm_<xyz>"` of
    the Haero-provided `nvcc_wrapper` (`haero_install/bin/nvcc_wrapper`).
  - We follow up with a quick `grep` to confirm this.

### Tokens

- AT2 requires 2 fine-grained tokens for the **mam4xxSNL** account from the
  `eagles-project` GitHub Organization in order to access information related
  to the `mam4xx` repo.
  - One token used to fetch and read/write runner information.
    - **Expires 11 April 2026**
  - One token used fetch and read repository information via the API.
    - **Expires 2 May 2025**

## TODO

- [x] Update job control section of README after the behavior is made clear.
  - @mjschmdt271
- [ ] Include a script to generate plots from within testing container?
  - @jaelynlitz?
- [ ] Unify all CI into a single top-level yaml file that calls the sub-cases.
  - This should provide finer control over what runs and when.
- [ ] Add testing for AMD GPUs on `caraway`.

### Low-priority

- [ ] Add CPU testing on `mappy` because "heck, why not?"
