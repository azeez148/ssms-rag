#!/bin/sh
set -e

# Startup behavior: migrations run to completion before the app starts
# accepting traffic — an ECS task (or a compose container) that passes its
# health check has always already got the schema it needs, never a stale one.
#
# The DB being reachable at the network level doesn't mean it's accepting
# connections yet (a fresh RDS instance, or a compose Postgres still running
# its own init scripts), so retry with backoff instead of failing on the
# first attempt.
#
# RUN_MIGRATIONS=false skips this step. With several ECS tasks starting at once
# (rolling deploy, autoscaling) they'd all race to migrate the same database;
# in that setup run migrations once as a separate one-off task
# (`command: ["alembic","upgrade","head"]`) and set RUN_MIGRATIONS=false on the
# service. Default stays true so a single container / compose just works.
if [ "${RUN_MIGRATIONS:-true}" != "true" ]; then
    echo "RUN_MIGRATIONS=${RUN_MIGRATIONS} — skipping alembic upgrade head"
    exec "$@"
fi

attempt=1
max_attempts=10
until alembic upgrade head; do
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "alembic upgrade head failed after $max_attempts attempts — giving up" >&2
        exit 1
    fi
    echo "alembic upgrade head failed (attempt $attempt/$max_attempts) — retrying in $((attempt * 2))s" >&2
    sleep $((attempt * 2))
    attempt=$((attempt + 1))
done

exec "$@"
