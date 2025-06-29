from cryptography.fernet import Fernet


class SecretVault:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, secret: str) -> bytes:
        return self.cipher.encrypt(secret.encode())

    def decrypt(self, token: bytes) -> str:
        return self.cipher.decrypt(token).decode()
