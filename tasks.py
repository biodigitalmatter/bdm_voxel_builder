from invoke import task


@task
def format(c):
    c.run("ruff check --fix .")
    c.run("ruff format .")


@task
def lint(c):
    c.run("ruff check .")

@task
def test(c):
    c.run("pytest")
