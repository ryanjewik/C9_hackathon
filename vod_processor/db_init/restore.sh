#!/bin/bash
# Restore the backup.sql database
# This script handles binary pg_dump format (.backup or custom format)

set -e

BACKUP_FILE="/docker-entrypoint-initdb.d/backup.sql"
DB_NAME="${POSTGRES_DB:-cloud9}"

echo "Checking backup file format..."

# Enable pg_trgm extension for fuzzy player name matching
echo "Enabling pg_trgm extension..."
psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" || true

# If a schema file is provided, apply it first (idempotent). This helps when
# the provided backup is data-only or pg_restore doesn't recreate schema.
SCHEMA_FILE="/docker-entrypoint-initdb.d/schema.sql"
if [ -f "$SCHEMA_FILE" ]; then
    echo "Schema file $SCHEMA_FILE found — applying before restore..."
    psql --username="$POSTGRES_USER" --dbname="$DB_NAME" -f "$SCHEMA_FILE" || true
    echo "Schema apply (pre-restore) attempted."
fi

# Check if the file is a custom format pg_dump
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
    echo "Attempting SQL script restore..."
    psql --username="$POSTGRES_USER" --dbname="$DB_NAME" < "$BACKUP_FILE" || true
    echo "SQL restore attempted."
fi

echo "Database restoration complete."
