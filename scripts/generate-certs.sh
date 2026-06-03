#!/bin/sh
set -e

SSL_DIR="${SSL_DIR:-/etc/nginx/ssl}"
mkdir -p "$SSL_DIR"

if [ ! -f "$SSL_DIR/server.crt" ] || [ ! -f "$SSL_DIR/server.key" ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$SSL_DIR/server.key" \
        -out "$SSL_DIR/server.crt" \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo "Self-signed certs generated."
else
    echo "Certs already exist."
fi
