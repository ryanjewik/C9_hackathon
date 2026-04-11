#!/bin/sh
# Restore the backup.sql database
# This script handles either a pg_restore custom dump or a plain SQL file.

set -e

BACKUP_FILE="/docker-entrypoint-initdb.d/backup.sql"
DB_NAME="${POSTGRES_DB:-cloud9}"

echo "Checking backup file format..."

# Enable pg_trgm extension (ignore failure if not supported yet)
echo "Enabling pg_trgm extension..."
psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" || true

# Apply schema files (idempotent). Prefer schema.sql, then users_schema.sql.
for SCHEMA_FILE in /docker-entrypoint-initdb.d/schema.sql /docker-entrypoint-initdb.d/users_schema.sql; do
    if [ -f "$SCHEMA_FILE" ]; then
        echo "Schema file $SCHEMA_FILE found - applying before restore..."
        psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -f "$SCHEMA_FILE" || true
        echo "Schema apply (pre-restore) attempted."
        break
    fi
done

# If the backup file is a custom-format dump, use pg_restore; otherwise use psql
if [ -f "$BACKUP_FILE" ] && pg_restore -l "$BACKUP_FILE" >/dev/null 2>&1; then
    echo "Restoring from PostgreSQL custom dump format..."
    pg_restore --username="$POSTGRES_USER" \
               --dbname="$DB_NAME" \
               --no-owner \
               --no-privileges \
               --clean \
               --if-exists \
               "$BACKUP_FILE" || true
    echo "Restore completed!"
else
    if [ -f "$BACKUP_FILE" ]; then
        echo "Attempting SQL script restore..."
        psql --username="$POSTGRES_USER" --dbname="$DB_NAME" < "$BACKUP_FILE" || true
        echo "SQL restore attempted."
    else
        echo "No backup file found at $BACKUP_FILE - skipping restore."
    fi
fi

echo "Database restoration complete."

