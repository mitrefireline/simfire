# Introduction

The Rothermel Fire Modeler uses [PyGame](https://www.pygame.org/wiki/about) to display and simulate the Rothermel Surface Fire Spread Model described in [this](https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf) paper.

[GitLab Page](https://gitlab.mitre.org/fireline/rothermel-modeling)

## Running the Simulation

Clone the repository:

```shell
git clone git@gitlab.mitre.org:fireline/rothermel-modeling.git
```

Then, install the requirements:

```shell
pip install -r requirements.txt
```

And run the `game_rothermal.py` script:

```shell
python game_rothermel.py
```

## Installing the Package

To use the package `rothsim` without cloning the repo, you must [create a GitLab Personal Access Token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#create-a-personal-access-token) and be added to the `fiReLine` GitLab group and/or the `fiReLine/rothermel-modeling` project. You'll want to make sure that the [personal access token scope](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#personal-access-token-scopes) is set to **`read_api`**. Once this has been created, you should copy the token and keep it in a secure location.

Then, you can install the `rothsim` package with the following command, making sure to replace `<your_personal_token>` with the token you just copied.

```shell
pip install rothsim --extra-index-url https://__token__:<your_personal_token>@gitlab.mitre.org/api/v4/projects/34582/packages/pypi/simple
```

For information on how to use the package, for now, use [`game_rothermel.py`](https://gitlab.mitre.org/fireline/rothermel-modeling/-/blob/master/game_rothermel.py) as a jumping-off point. And to configure simulation, go to the [Configuring the Simulation](config.md) page.

### Modifying `.bashrc` for Easy Install

If you'd like, you can modify your `.bashrc` (or `.zshrc`, depending on your terminal) file to more easily install the package without the long URL.

Add the following to your `.bashrc`, making sure to substitute your own `GITLAB_READ_API_TOKEN` from the section above:

```shell
# GitLab Registry Access
export GITLAB_READ_API_TOKEN=<your read api token>

pip_install_rothsim () {
    version=$1
    if [ -n "$version" ]; then
        install_statement="pip install rothsim==${version} --extra-index-url https://__token__:${GITLAB_READ_API_TOKEN}@gitlab.mitre.org/api/v4/projects/34582/packages/pypi/simple"
    else
        install_statement="pip install rothsim --extra-index-url https://__token__:${GITLAB_READ_API_TOKEN}@gitlab.mitre.org/api/v4/projects/34582/packages/pypi/simple"
    fi
    eval "${install_statement}"
}
```

Now you can install `rothsim` by calling the following from your terminal

```shell
pip_install_rothsim
```

And you can supply a version by adding a version number

```shell
pip_install_rothsim <version number>

```

By default, it will get the most recent version. Right now, whatever is in `master` will always be version `0.0.0`. The other versions can be seen in the [Releases Section of the GitLab](https://gitlab.mitre.org/fireline/rothermel-modeling/-/releases) or on the [Package Registry Page](https://gitlab.mitre.org/fireline/rothermel-modeling/-/packages).
