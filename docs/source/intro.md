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
