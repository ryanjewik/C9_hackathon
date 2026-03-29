#!/bin/bash
# Restore the backup.sql database
# This script handles binary pg_dump format (.backup or custom format)

set -e

BACKUP_FILE="/backup/backup.dump"
DB_NAME="${POSTGRES_DB:-cloud9}"

echo "Checking backup file format..."

# Enable pg_trgm extension for fuzzy player name matching
echo "Enabling pg_trgm extension..."
psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" || true

# If a schema file is provided, apply it first (idempotent). This helps when
# the provided backup is data-only or pg_restore doesn't recreate schema.
# Support either schema.sql or users_schema.sql placed in the /backup folder.
for SCHEMA_FILE in /backup/schema.sql /backup/users_schema.sql; do
    if [ -f "$SCHEMA_FILE" ]; then
        echo "Schema file $SCHEMA_FILE found — applying before restore..."
        psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -f "$SCHEMA_FILE" || true
        echo "Schema apply (pre-restore) attempted."
        break
    fi
done

# Check if the backup file is a pg_restore-compatible archive. Some minimal
# postgres images don't include the `file` utility, so use `pg_restore -l` to
# probe the archive format instead.
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
elif [ -f "$BACKUP_FILE" ]; then
    echo "Attempting SQL script restore (plain SQL file)..."
    psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -f "$BACKUP_FILE" || true
    echo "SQL restore attempted."
else
    echo "No backup file found at $BACKUP_FILE; skipping data restore."
fi

echo "Database restoration complete."
