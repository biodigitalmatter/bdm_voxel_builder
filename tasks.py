from invoke import task


@task
def format(c):
    c.run("ruff check --fix .")
    c.run("ruff format .")


@task
def lint(c):
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
