from invoke import task


@task
def format(c):
    c.run("ruff check --fix .", warn=True)
    c.run("ruff format .")


@task
def check(c):
    c.run("ruff check .")


@task
def test(c, verbose=False):
    cmd = "pytest"
    if verbose:
        cmd += " -v"
    c.run(cmd)


@task
def install(c):
    c.run("pip install -e .")


@task
def start_ppl(c):
    image = "ghcr.io/biodigitalmatter/ros:main"
    port = 9090
    cmd = "roslaunch biodigitalmatter_ros planning.launch"
    c.run(f"docker run -p {port}:{port} {image} {cmd}")
