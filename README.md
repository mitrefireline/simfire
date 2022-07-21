# SimFire Fire Simulator

<p align="center">
    <img src="assets/icons/rl_logo_horizontal.png">
</p>

## Introduction

SimFire uses [PyGame](https://www.pygame.org/wiki/about) to display and simulate different fire spread models, including the Rothermel Surface fire spread model described in [this](https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf) paper.

For more comprehensive documentation, go to our [docs page](https://fireline.pages.mitre.org/simfire).


## Running the Simulation
<figure>
    <p align="center">
        <img src="assets/gifs/simulation_33.06N_116.58W.gif" width="225" />
        <img src="assets/gifs/simulation_39.67N_119.80W.gif" width="225" />
    </p>
    <figcaption align = "center"><b>Left: Fire simulated near Julian, CA. Right: Fire simulated near Reno, NV.
                                    <br>Both fires have winds from the east at 20mph<b></figcaption>
</figure>

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

### Running as a Python Module

```python
from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation

config = Config("configs/operational_config.yml")
sim = FireSimulation(config)

# Run a 1 hour simulation
sim.run("1h")

# Run the same simulation for 30 more minutes
sim.run("30m")

# Render the next 2 hours of simulation
sim.rendering = True
sim.run("2h")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
sim.save_gif()
sim.save_spread_graph()
# Saved to the location specified in the config: simulation.save_path

# Turn off rendering
sim.rendering = False
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


## Contributing

For contributing, see the [Contribution Page](https://fireline.pages.mitre.org/simfire/contributing.html) in our [docs](https://fireline.pages.mitre.org/simfire).
