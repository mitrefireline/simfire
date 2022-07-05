# Introduction

SimFire uses [PyGame](https://www.pygame.org/wiki/about) to display and simulate different fire spread models, including the Rothermel Surface fire spread model described in [this](https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf) paper.

[GitLab Page](https://gitlab.mitre.org/fireline/simfire)

## Running the Simulation

Clone the repository:

```shell
git clone git@gitlab.mitre.org:fireline/simfire.git
```

Then, install the requirements:

```shell
pip install poetry
poetry install --no-dev
```

And run the `run_game.py` script:

```shell
python run_game.py
```

## Installing the Package

### Installing from Artifactory

To do this, you must be on the MITRE network and have access to [artifacts.mitre.org](https://artifacts.mitre.org).

First, you should change or create a `pip.conf` file in your `$HOME/.pip/` directory (taken from [this confluence page](https://confluence.ecis.mitre.org/pages/viewpage.action?spaceKey=ETC&title=Artifactory+Pro+-+artifacts.mitre.org#ArtifactoryProartifacts.mitre.org-Python:pip)):

```
[global]
index-url = https://artifacts.mitre.org/artifactory/api/pypi/python/simple
timeout = 60
# some users have reported needing the following on Windows
trusted-host = artifacts.mitre.org
[search]
index = https://artifacts.mitre.org/artifactory/api/pypi/python
```

When save your `pip.conf` file with this configuration, just run the following command to install `simfire`:

```shell
pip install simfire
```

And for a specific version:

```shell
pip install simfire==<version>
```

### Installing from GitLab PyPi Registry

To use the package `simfire` without cloning the repo, you must [create a GitLab Personal Access Token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#create-a-personal-access-token) and be added to the `fiReLine` GitLab group and/or the `fiReLine/simfire` project. You'll want to make sure that the [personal access token scope](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#personal-access-token-scopes) is set to **`read_api`**. Once this has been created, you should copy the token and keep it in a secure location.

Then, you can install the `simfire` package with the following command, making sure to replace `<your_personal_token>` with the token you just copied.

```shell
pip install simfire --extra-index-url https://__token__:<your_personal_token>@gitlab.mitre.org/api/v4/projects/34582/packages/pypi/simple
```

For information on how to use the package, for now, use [`run_game.py`](https://gitlab.mitre.org/fireline/simfire/-/blob/main/run_game.py) as a jumping-off point. And to configure simulation, go to the [Configuring the Simulation](config.md) page.

#### Modifying `.bashrc` for Easy GitLab PyPi Registry Install

If you'd like, you can modify your `.bashrc` (or `.zshrc`, depending on your terminal) file to more easily install the package without the long URL.

Add the following to your `.bashrc`, making sure to substitute your own `GITLAB_READ_API_TOKEN` from the section above:

```shell
# GitLab Registry Access
export GITLAB_READ_API_TOKEN=<your read api token>

pip_install_simfire () {
    version=$1
    if [ -n "$version" ]; then
        install_statement="pip install simfire==${version} --extra-index-url https://__token__:${GITLAB_READ_API_TOKEN}@gitlab.mitre.org/api/v4/projects/34582/packages/pypi/simple"
    else
        install_statement="pip install simfire --extra-index-url https://__token__:${GITLAB_READ_API_TOKEN}@gitlab.mitre.org/api/v4/projects/34582/packages/pypi/simple"
    fi
    eval "${install_statement}"
}
```

Now you can install `simfire` by calling the following from your terminal

```shell
pip_install_simfire
```

And you can supply a version by adding a version number

```shell
pip_install_simfire <version number>
```

By default, it will get the most recent version. Right now, whatever is in `main` will always be version `0.0.0`. The other versions can be seen in the [Releases Section of the GitLab](https://gitlab.mitre.org/fireline/simfire/-/releases) or on the [Package Registry Page](https://gitlab.mitre.org/fireline/simfire/-/packages).


## Setting Up xpra for Remote Simulation Visualization

If you'd like, you can modify your `.bashrc` (or `.zshrc`, depending on your terminal) file to easily forward the simulation display from your remote machine.

Before adding to the `.bashrc`, install the following packages on your remote terminal, which will open a persistent tmux session titled `xpra` to forward your remote ports:

```shell
sudo apt install xpra
sudo apt install tmux
```

Add the following to your `.bashrc`, making sure to substitute your own `<DISPLAY>` from the section above:

```shell

# Added for XPRA support
if ! tmux has-session -t xpra
then
    tmux new -s xpra -d
    tmux send -t xpra.0 'xpra start :<DISPLAY> --systemd-run=no --daemon=no' Enter
fi

```

Finally, on your local terminal, install the `xpra` package and then run the following command:

```shell
xpra attach --ssh=ssh ssh://<CONTAINER>@rlord/<DISPLAY>
```
