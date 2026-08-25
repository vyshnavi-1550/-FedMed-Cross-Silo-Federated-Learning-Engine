"""
Week 2 — Generate Self-Signed TLS Certificates (for local development/testing)

Creates a Certificate Authority (CA) and a server certificate signed by it,
which Flower's gRPC server/client can use to encrypt traffic between the
central server and hospital nodes.

IMPORTANT: self-signed certs are fine for local testing and coursework
demos. A REAL deployment across actual separate hospitals would need
certificates from a trusted CA (or an internally-trusted PKI setup) --
self-signed certs are not appropriate for production healthcare systems.

Requires OpenSSL (usually pre-installed on Mac/Linux; on Windows, Git Bash
includes it, or install via https://slproweb.com/products/Win32OpenSSL.html)

Run:
    python generate_certs.py
"""

import subprocess
import os

CERTS_DIR = os.path.join(os.path.dirname(__file__), "certs")


def find_openssl_config():
    """Anaconda's bundled OpenSSL on Windows often can't find its own config
    file automatically. Search common Anaconda locations and set OPENSSL_CONF
    so the openssl commands below don't fail."""
    candidates = [
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "ssl", "openssl.cnf"),
        os.path.join(os.environ.get("USERPROFILE", ""), "anaconda3", "Library", "ssl", "openssl.cnf"),
        os.path.join(os.environ.get("USERPROFILE", ""), "anaconda3", "ssl", "openssl.cnf"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            print(f"Found OpenSSL config at: {path}")
            os.environ["OPENSSL_CONF"] = path
            return
    print("WARNING: could not auto-locate openssl.cnf. If commands fail below, "
          "search your Anaconda install for 'openssl.cnf' and set OPENSSL_CONF manually.")


def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def main():
    find_openssl_config()
    os.makedirs(CERTS_DIR, exist_ok=True)
    os.chdir(CERTS_DIR)

    print("Generating a local Certificate Authority (CA)...")
    run('openssl genrsa -out ca.key 4096')
    run(
        'openssl req -new -x509 -key ca.key -sha256 -subj "/C=IN/O=FedMed/CN=FedMed Local CA" '
        '-days 365 -out ca.crt'
    )

    print("Generating the server's private key and certificate signing request...")
    run('openssl genrsa -out server.key 4096')
    run(
        'openssl req -new -key server.key -out server.csr '
        '-subj "/C=IN/O=FedMed/CN=localhost"'
    )

    print("Signing the server certificate with our local CA...")
    # SAN (Subject Alternative Name) for localhost -- required by modern TLS clients
    with open("server_ext.cnf", "w") as f:
        f.write("subjectAltName=DNS:localhost,IP:127.0.0.1\n")

    run(
        'openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial '
        '-out server.pem -days 365 -sha256 -extfile server_ext.cnf'
    )

    print(f"\nDone. Certificates created in: {CERTS_DIR}")
    print(" - ca.crt      (Certificate Authority -- clients use this to verify the server)")
    print(" - server.key  (server's private key -- keep secret, server-side only)")
    print(" - server.pem  (server's certificate -- presented to clients)")


if __name__ == "__main__":
    main()
