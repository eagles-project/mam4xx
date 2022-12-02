# PNNL CI Documentation for mam4xx
This is used to track any maintanence information for PNNL CI. We will also track any current TODOs/notes for developers.
## Usage
We currently only have 2 k8s runners, and so you must only run 2 concurrent pipelines at a time.

Since we are only able to target one Kokkos CUDA arch at a time https://github.com/kokkos/kokkos/issues/4953, we can only run CI on one Deception partition.

**NOTE: Since we require CI to run on only one partition, CI might not always go through if Deception partition `dl_shared` is particularly busy.**

### Skipping CI runs:
PNNL CI will only run when you are adding new commits to an existing merge request.

You can add `[skip-ci]` in order to prevent CI jobs from running at PNNL. TODO involves adding support for skipping CI when certain tags are present in a PR.

#### TODO:
- [ ] Use installed HAERO in project share to avoid re-building each time
- [ ] Add support for a variety of paritions on Deception. We currently only target dl_shared as we can only choose one cuda arch.
- [ ] Add way to skip CI using a GitHub tag in both GitLab and GitHub
- [ ] Port pipeline to AMD architectures

#### Done:
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

## Access Token
@CameronRutherford currently maintains the access token used to enable GitHub mirroring. 
This token is set to expire in one year, and someone will need to ensure that this integration is renewed each year.

## PNNL Site Config
We have manually configured PNNL CI to point to the YAML file in `/.github/pnnl-ci/pnnl.gitlab-ci.yml`. Make sure to re-configure this when refreshing connection.

## GitHub/GitLab Integration
You need to generate a Personal Access Token (PAT) through GitHub project before starting this process.

We will also be creating a pull mirror, and not a push mirror, as pull mirrors are updated based on an external source, whereas push mirrors are preferred when pushing updates externally.

Steps:
1. Create an empty project in GitLab. **DO NOT** initialize using in-build GitHub integration, as this is broken for running pipelines.
1. Set up repository mirroring, ensuring you enable running pipelines and keeping divergent refs. Use GitHub username within `https` URL and Personal Access token as the password. `https://CameronRutherford@github.com/eagles-project/mam4xx` would be the repository URL in this case, where my username can be replaced with the account where the PAT was generated, and you would place the PAT in the field which asks for a password. End results should be a repository mirror with `https://*****:*****@github.com/eagles-project/mam4xx` showing as the URL.
1. Enable the GitHub integration in Settings > Integrations in GitLab. This will post pipeline status to the relevant Pull Requests, and you will need to add a personal access token used here as well.
1. Ensure your YAML has correct syntax, and you should be good to go! Since we mirror everything in the repo, we will have copies of issues and PRs as well.

Since this integration is automatically configured through GitLab premium, pipeline status will automatically be posted to commits/PRs.

There is a way to orchestrate this pipeline posting through non-premium GitLab as well - https://ecp-ci.gitlab.io/docs/guides/build-status-gitlab.html

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

Since we are installing in GitLab pipelines, this script uses the variable `CI_HTTPS_INSTALL` in order to force the `build-haero.sh` script to clone submodules using HTTPS instead of SSH.

It does this by manually find and replacing the `.gitmodules` files in each repository where relevant with `https://.../` instead of `git@...:`.

Additionally, for some reason `SYSTEM_NAME` is configured on PNNL login nodes, but when running in a job this variable proves unhelpful. As such, we export `SYSTEM_NAME=deception` in this script before running.
