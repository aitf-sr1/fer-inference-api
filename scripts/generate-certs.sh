#!/bin/sh
set -e

SSL_DIR="${SSL_DIR:-/etc/nginx/ssl}"
DOMAIN="${DOMAIN:-localhost}"
mkdir -p "$SSL_DIR"

if [ ! -f "$SSL_DIR/server.crt" ] || [ ! -f "$SSL_DIR/server.key" ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$SSL_DIR/server.key" \
        -out "$SSL_DIR/server.crt" \
        -subj "/CN=${DOMAIN}" \
        -addext "subjectAltName=DNS:${DOMAIN},DNS:localhost"
    echo "Self-signed certs generated for ${DOMAIN}."
else
    echo "Certs already exist."
fi
