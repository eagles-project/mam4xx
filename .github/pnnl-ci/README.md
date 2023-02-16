# PNNL CI Documentation for mam4xx
This is used to track any maintanence information for PNNL CI. We will also track any current TODOs/notes for developers.

## Usage
We currently only have 2 k8s runners, and so you must only run 2 concurrent pipelines at a time.

Since we added resource groups to control how many pipelines can run at once, we will use 2 runners in a given pipeline stage.

If another pipeline is running, you will have to wait for it to finish before yours can start.

CI runs on a variety of hardward architectures by using a perl script when the job is run to select the right partition:

The possible  GPU architectures are:

| GPU Type | Cuda Architecture | SLURM partition |
|-|-|-|
| P100 | 60 | dl/dl_shared |
| V100 | 70 | dlv/dlv_shared |
| RTX 2080 Ti | 75 | dlt/dlt_shared |

For some reason only the HPC runners are configured to run at the moment, and so all stages will share that base configuration.
## Rebuilding HAERO in CI

You can either add `[hearo-rebuild]` or `[rebuild-haero]` directly to your commit message, or go to https://code.pnnl.gov/e3sm/eagles/mam4xx and trigger the rebuild pipeline manually once you have pushed to your branch.

Make sure you either push to GitHub and have the mirror update first, or just push to the GitLab directly.

### Skipping CI runs:
PNNL CI will only run when you are adding new commits to an existing merge request.

You can add `[skip-ci]` in order to prevent CI jobs from running at PNNL. TODO involves adding support for skipping CI when certain tags are present in a PR.

#### TODO:
- [ ] Port pipeline to AMD architectures

#### Done:
- [x] Consider cleaning up old installations and adding permissions changes so all users can use shared installation
- [x] Add support for a variety of paritions on Deception. We currently only target dl_shared as we can only choose one cuda arch
- [x] Add way to skip CI using a GitHub tag in both GitLab and GitHub
- [x] Run CI based on commit message or manual trigger
- [x] Get mam4xx building for GPU locally, then get working in CI
- [x] Only run 2 jobs at a time as we only have 2 runners
- [x] Only run PNNL CI in PRs
- [x] Refactor CI YAML to remove duplication across scripts
- [x] Support full matrix of build types (single, double etc.)
- [x] Rebuild HAERO in manual pipeline
- [x] Add CMake / ctest configuration in CI
- [x] Add way to skip CI using a commit message
- [x] Add support for cloning with ssh in CI, with documentation
- [x] Build HEARO without cloning mam4xx in CI step
- [x] Ensure that pipelines are not false positive/negative
- [x] Streamline CI rebuilding of HAERO to happen with one button (need to work around 2 max job limit)
- [x] Use installed HAERO in project share to avoid re-building each time

## Access Token
@CameronRutherford currently maintains the access token used to enable GitHub mirroring. 
This token is set to expire in one year, and someone will need to ensure that this integration is renewed each year.

https://code.pnnl.gov/help/user/project/repository/mirror/pull.md - make sure that when you create the PAT for this integration, that you use `write` as the scope, so that GitHub actions can write to the PNNL GitLab with updates as they are added.

## Pipeline Trigger Token

You will need to setup up a pipeline trigger token in order to allow GitHub acitons to trigger CI pipelines.

## PNNL Site Config
We have manually configured PNNL CI to point to the YAML file in `/.github/pnnl-ci/pnnl.gitlab-ci.yml`. Make sure to re-configure this if you need to re-configure the repository.

## GitHub/GitLab Integration
You need to generate a Personal Access Token (PAT) through GitHub project before starting this process per the section above.

We are going to set up a push mirror that is updated with each pull request update. Through a GitHub action, each check will:
1. Push updates to all branches in the GitLab
1. Trigger a pipeline to run using the Pipeline Trigger token
1. PNNL GitLab will post a new message describing pipeline status in a separate check

The GitHub action in `/.github/workflows/pnnl_push_mirror.yml` relies on the following GitHub secrets. Make sure to configure these if they are expired/broken:
1. GITLAB_ACCESS_TOKEN : This is the PAT configured with write permissions for the push mirror action
1. GITLAB_PIPELINE_TRIGGER_TOKEN : This is a separate token that allows you to use the pipeline trigger API
1. GITLAB_REPO_URL : The same url that one would use for adding mam4xx GitLab as a remote w/ https connection
1. GITLAB_USER : The username to associate with push mirror actions (can be any valid user)

In order to set this up:
1. Create an empty project in GitLab. **DO NOT** initialize using in-build GitHub integration, as this is broken for running pipelines.
1. Enable the GitHub integration in Settings > Integrations in GitLab. This will post pipeline status to the relevant Pull Requests, and you will need to add a personal access token used here as well.
1. Ensure your YAML has correct syntax, and you should be good to go!

Since the pipeline status is automatically configured through GitLab premium + GitHub integration, pipeline status will automatically be posted to commits/PRs.

There is a way to orchestrate this pipeline posting through non-premium GitLab as well - https://ecp-ci.gitlab.io/docs/guides/build-status-gitlab.html...

## Scripts
There are shared environment variables that are propogated across both scripts, and each job shares the same template in order to reduce code duplication.

The shared variables are:
- `HAERO_INSTALL` - specifying where haero is/should be installed
- `BUILD_TYPE` - Debug/Release
- `PRECISION` - Single/Double, only applies to haero build stage

### `ci.sh`
Used to build and test mam4xx in CI using HAERO installed in project share.

Similar to the `rebuild-haero.sh` script, since we are building in CI, SSH submodules will not suffice. As such this scripts clones the validation repo manually after applying a perl script on the `.gitmodules` file.

### `rebuild-haero.sh`
Used to re-configure HAERO in project share, along with configuring permissions so other users can configure with shared installation.

Since we are installing in GitLab pipelines, we are unable to clone with SSH. This requirement resulted in a separate script for CI, where HTTPS is used for submodules instead of SSH.

It does this by manually find and replacing the `.gitmodules` files in each repository where relevant with `https://.../` instead of `git@...:`.

Additionally, for some reason `SYSTEM_NAME` is configured on PNNL login nodes, but when running in a job this variable proves unhelpful. As such, we export `SYSTEM_NAME=deception` in this script before running.
