# Trainer Actions

The intent of the modules in this directory are to provide a fully detached way to
achieve very specific tasks with Trainer. Notebooks are often tricky to get working
for long running jobs due to various issues that can cause the kernel to die.

The general goals of these actions are:

- Purpose built Python modules that achieve a very specific task
- Defining all inputs should be a function of environment variables i.e. with a `.env` file
- The actions can be run in containers for portability

# Container Build

To build the container run:

```
docker build -t gretelai/trainer .
```

The `data` directory can be used as a volume mount into the container, it currently
contains an example Gretel Config to be used with a demo DB.

Within the container, we have a `/gretel/data` directory that should be used to mount
to a directory on the host.

# Actions

The only requirements to run an action should be:

- An `.env` file
- Execution of the action's Python module. If using the Docker container, you must provide the `GRETEL_ACTION` variable, this should match the name of the Python module (action) you want to run.

## Transform Relational

This action consumes a single Gretel Config, two connections to databases (source and sink), Gretel API key and executes a full database transform
using the provided config. An example `.env` is below.

```
GRETEL_ACTION=transform_relational
GRETEL_CONFIG=/gretel/data/transform_config.yml
GRETEL_API_KEY=grtuXXX
GRETEL_PROJECT_DISPLAY_NAME=transform-relational-action
SOURCE_DB=mysql://dynf4plkdpg6e14abhf5:PASSWORD@aws.connect.psdb.cloud:3306/relational-demo?ssl=true&charset=utf8mb4
SINK_DB=sqlite:////gretel/data/transformed.db
WEBHOOK=https://hooks.slack.com/XXX
```

Then you can launch the container:

```
docker run -it --rm -v $PWD/data:/gretel/data --env-file .env gretelai/trainer
```

You can add the `-d` flag to run detached, which gives you a headless execution.
