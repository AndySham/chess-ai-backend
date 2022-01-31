
# Build Steps

The following build procedure is for Ubuntu Linux machines.

## Set up PostgreSQL

### Local Install

Refer to the following link to [https://www.postgresqltutorial.com/install-postgresql-linux/](install PostgreSQL for Linux).

Then set up a PostgreSQL user with the appropriate permissions
```
$ sudo -i -u postgres   # act as the postgres linux user
$ psql                  # enter the postgres shell
postgres=# CREATE DATABASE psqldb;
                        # create a database. psqldb can be changed
postgres=# CREATE USER psqluser WITH PASSWORD 'psqlpw'; 
                        # create a username and password. psqluser and psqlpw can be changed
postgres=# GRANT ALL PRIVILIGES ON DATABASE "psqldb" TO psqluser;
                        # allow your user to modify the database freely
postgres=# \q           # exit the postgres shell
$ exit                  # return to original linux user
```

Then install `libpq` to allow the `psycopg2` package to access your local PostgreSQL daemon.
```
$ sudo apt install libpq-dev
```

## Create Environment

First, we install `python3-venv` to virtualise the project environment. In Ubuntu, one must run
```
$ sudo apt install python3-venv
```
Install the `virtualenv` pip package if you haven't already.
```
$ pip3 install virtualenv
```
Then create a venv module in the local folder.
```
$ python3 -m venv venv
```
Then, we enter the virtual environment by running the following.
```
$ source venv/bin/activate
```
Finally, in the venv, install all the dependencies.
```
$ pip install -r requirements.txt
```

## Updating dependencies

If in the venv you `pip install` a new package, remember to pipe the updated dependency list to `requirements.txt` by running the following.
```
$ pip freeze | tee requirements.txt
```

# Commands

