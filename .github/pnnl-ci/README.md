# PNNL CI Documentation for mam4xx

This is used to track any maintanence information for PNNL CI. We will also track any current TODOs/notes for developers.

## TODO

- [ ] Rebuild HAERO in manual pipeline
- [ ] Use installed HAERO in project share to avoid re-building each time
- [ ] Add CMake / ctest configuration in CI
- [ ] Ensure that pipelines are not false positive/negative
- [ ] Port pipeline to AMD architectures

## Access Token

@CameronRutherford currently maintains the access token used to enable GitHub mirroring. This token is set to expire in one year, and someone will need to ensure that this integration is renewed each year.

## PNNL Site Config

We have manually configured PNNL CI to point to the YAML file in `/.github/pnnl-ci/pnnl.gitlab-ci.yml`. Make sure to re-configure this when refreshing connection.

## GitHub configuration

https://forum.gitlab.com/t/duplicate-pipelines-for-ci-cd-for-external-repo-github/32551/4 - I had to enable these features in GitHub settings in order to avoid duplicate pipelines being triggered.

The above might not be true, but I needed to give the Personal Access Token additional permission per https://docs.gitlab.com/ee/ci/ci_cd_for_external_repos/github_integration.html#connect-with-personal-access-token

## Scipts

### `ci.sh`

Used to build and test mam4xx in CI using HAERO installed in project share.

### `rebuild-haero.sh`

Used to re-configure HAERO in project share, along with configuring permissions so other users can configure with shared installation.