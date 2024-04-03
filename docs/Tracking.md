# Experiment Tracking

## Comet.ml

* We recommend using [Comet.ml](https://www.comet.ml/) for experiment tracking.
* To activate Comet.ml,  head over to the [Comet.ml website](https://www.comet.ml/) and create an account.
* Once you have created an account, you can create a new project and get your API key.
* Set the `COMET_API_KEY` environment variable to your API key. We recommend adding this to your `.bashrc` or `.bash_profile` file. e.g.
```bash
# .bashrc
export COMET_API_KEY="your-api-key"
```