# PNNL CI Documentation for mam4xx

This is used to track any maintanence information for PNNL CI. We will also track any current TODOs/notes for developers.

## TODO

- [ ] Add way to skip CI using a commit message
- [ ] Only run GPU CI in PRs
- [ ] Rebuild HAERO in manual pipeline
- [ ] Use installed HAERO in project share to avoid re-building each time
- [ ] Add CMake / ctest configuration in CI
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

1. Import an existing GitHub repository in GitLab using mirroring. Make sure to add your username, and use PAT as password auth
1. Enable the GitHub integration in Settings > Integrations in GitLab using PAT
1. Ensure your YAML has correct syntax, and you should be good to go!

Since this integration is automatically configured through GitLab premium, pipeline status will automatically be posted to commits/PRs.

## Scipts

### `ci.sh`

Used to build and test mam4xx in CI using HAERO installed in project share.

### `rebuild-haero.sh`

Used to re-configure HAERO in project share, along with configuring permissions so other users can configure with shared installation.

