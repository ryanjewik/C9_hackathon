#!/bin/bash
# Restore the backup.sql database
# This script handles binary pg_dump format (.backup or custom format)

set -e

BACKUP_FILE="/docker-entrypoint-initdb.d/backup.sql"
DB_NAME="${POSTGRES_DB:-cloud9}"

echo "Checking backup file format..."

# Check if the file is a custom format pg_dump
if file "$BACKUP_FILE" | grep -q "PostgreSQL custom database dump"; then
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
