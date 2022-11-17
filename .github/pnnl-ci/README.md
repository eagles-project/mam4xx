# PNNL CI Documentation for mam4xx

This is used to track any maintanence information for PNNL CI. We will also track any current TODOs/notes for developers.

## TODO

- [x] Only run PNNL CI in PRs
- [ ] Add support for a variety of paritions on Deception
- [ ] Build HEARO without cloning mam4xx in CI step
- [x] Refactor CI YAML to remove duplication across scripts
- [ ] Add way to skip CI using a commit message
- [x] Support full matrix of build types (single, double etc.)
- [x] Rebuild HAERO in manual pipeline
- [x] Use installed HAERO in project share to avoid re-building each time
- [x] Add CMake / ctest configuration in CI
- [ ] Ensure that pipelines are not false positive/negative
- [ ] Port pipeline to AMD architectures

## Access Token

@CameronRutherford currently maintains the access token used to enable GitHub mirroring. 
This token is set to expire in one year, and someone will need to ensure that this integration is renewed each year.

## PNNL Site Config

We have manually configured PNNL CI to point to the YAML file in `/.github/pnnl-ci/pnnl.gitlab-ci.yml`. Make sure to re-configure this when refreshing connection.

## GitHub/GitLab configuration

You need to generate a Personal Access Token (PAT) through GitHub project before starting this process.

Steps:

1. Create an empty project in GitLab. **DO NOT** initialize using in-build GitHub integration, as this is broken for running pipelines.
1. Set up repository mirroring, ensuring you enable running pipelines and keeping divergent refs. Use GitHub username within `https` URL and Personal Access token as the password.
1. Enable the GitHub integration in Settings > Integrations in GitLab. This will post pipeline status, and should automatically detect PAT.
1. Ensure your YAML has correct syntax, and you should be good to go! Since we mirror everything in the repo, we will have copies of issues and PRs as well.

Since this integration is automatically configured through GitLab premium, pipeline status will automatically be posted to commits/PRs.

## Scipts

### `ci.sh`

Used to build and test mam4xx in CI using HAERO installed in project share.

### `rebuild-haero.sh`

Used to re-configure HAERO in project share, along with configuring permissions so other users can configure with shared installation.

