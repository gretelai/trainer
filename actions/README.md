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

# Example

This example will take a single Gretel Transform Config and apply it to a demo MySQL DB
running in Planet Scale and write the transformed data out to a SQLite db on disk.

First create a `.env` file (in this directory) with the following contents, you'll need to change:

- Gretel API Key
- The password for the MySQL db

```
GRETEL_ACTION=transform_relational
GRETEL_CONFIG=/gretel/data/transform_config.yml
GRETEL_API_KEY=grtuXXX
SOURCE_DB=mysql://dynf4plkdpg6e14abhf5:PASSWORD@aws.connect.psdb.cloud:3306/relational-demo?ssl=true&charset=utf8mb4
SINK_DB=sqlite:////gretel/data/transformed.db
```

Then you can launch the container:

```
docker run -it --rm -v $PWD/data:/gretel/data --env-file .env gretelai/trainer
```