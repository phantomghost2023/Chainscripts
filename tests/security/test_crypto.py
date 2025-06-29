import pytest
from chainscript.security import (
    encrypt_data,
    decrypt_data,
    sign_data,
    verify_signature,
    generate_key_pair,
)


@pytest.fixture
def key_pair():
    return generate_key_pair()


def test_generate_key_pair():
    public_key, private_key = generate_key_pair()
    assert public_key is not None
    assert private_key is not None


def test_encrypt_decrypt_data(key_pair):
    public_key, private_key = key_pair
    original_data = b"This is a secret message."
    encrypted_data = encrypt_data(original_data, public_key)
    decrypted_data = decrypt_data(encrypted_data, private_key)
    assert original_data == decrypted_data


def test_sign_verify_data(key_pair):
    public_key, private_key = key_pair
    data_to_sign = b"Data to be signed."
    signature = sign_data(data_to_sign, private_key)
    assert verify_signature(data_to_sign, signature, public_key)


def test_verify_signature_invalid(key_pair):
    public_key, private_key = key_pair
    data_to_sign = b"Data to be signed."
    wrong_data = b"Wrong data."
    signature = sign_data(data_to_sign, private_key)
    assert not verify_signature(wrong_data, signature, public_key)


def test_encrypt_data_invalid_key():
    with pytest.raises(ValueError):
        encrypt_data(b"data", "invalid_key")


def test_decrypt_data_invalid_key():
    with pytest.raises(ValueError):
        decrypt_data(b"data", "invalid_key")


def test_sign_data_invalid_key():
    with pytest.raises(ValueError):
        sign_data(b"data", "invalid_key")


def test_verify_signature_invalid_key():
    with pytest.raises(ValueError):
        verify_signature(b"data", b"signature", "invalid_key")
