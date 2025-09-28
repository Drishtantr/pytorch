import base64
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# Simulating the server (storing public keys and verifying signatures)


class MockServer:
    def __init__(self):
        self.stored_public_keys = {}  # Store public keys by username

    # Client sends public key to the server for registration
    def store_public_key(self, username, public_key_pem):
        self.stored_public_keys[username] = public_key_pem
        print(f"Public key registered for {username}")
        return {"status": "success", "message": f"Public key stored for {username}"}

    # Server authenticates client based on stored public key
    def authenticate(self, username, signature_b64, payload):
        if username not in self.stored_public_keys:
            return {"status": "failure", "message": "Username not registered"}

        public_key_pem = self.stored_public_keys[username]
        public_key = serialization.load_pem_public_key(
            public_key_pem, backend=default_backend()
        )

        signature = base64.b64decode(signature_b64)

        try:
            public_key.verify(
                signature,
                payload.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return {"status": "success", "message": "You're connected"}
        except Exception as e:
            return {"status": "failure", "message": f"Couldn't verify: {str(e)}"}

# Client-side simulation


def client_test():
    username = "alice"

    # Load private key from private.pem
    with open("private.pem", "rb") as f:
        private_key_pem = f.read()
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=default_backend()
        )

    # Load public key from public.pem
    with open("public.pem", "rb") as f:
        public_key_pem = f.read()

    # Simulate server
    server = MockServer()

    # Register public key with the server
    server.store_public_key(username, public_key_pem)

    # Prepare a payload
    payload = {"action": "get_data", "msg": "Hello Server!"}
    payload_bytes = json.dumps(payload).encode()

    # Sign the payload with the private key
    signature = private_key.sign(
        payload_bytes,
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    signature_b64 = base64.b64encode(signature).decode()

    tampered_payload = {"action": "get_data",
                        "msg": "Hello Hacker!"}  # changed message

    tampered_payload_str = json.dumps(tampered_payload)

    # Authenticate with the server
    auth_response = server.authenticate(
        username, signature_b64, json.dumps(payload))
    print("Authentication response:", auth_response)


if __name__ == "__main__":
    client_test()
