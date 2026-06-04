import os
import random
import pandas as pd
from Crypto.Cipher import AES, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad

SAMPLES_PER_CLASS = 500
PLAINTEXT_SIZE    = 256

def generate_realistic_plaintext(size: int) -> bytes:
    pattern = random.choice(['text', 'json', 'binary', 'mixed', 'random_heavy'])
    if pattern == 'text':
        words = [b'hello', b'world', b'test', b'data', b'info', b'message',
                 b'request', b'response', b'server', b'client', b'user', b'data']
        result = b''
        while len(result) < size:
            result += random.choice(words) + b' '
        return result[:size]
    elif pattern == 'json':
        keys = [b'name', b'value', b'type', b'id', b'status', b'code']
        result = b'{'
        while len(result) < size - 1:
            key = random.choice(keys)
            val = str(random.randint(0, 1000)).encode()
            result += key + b':' + val + b','
        return (result + b'}')[:size]
    elif pattern == 'binary':
        header = bytes([0x00, 0xFF, 0xAA, 0x55])
        return header + os.urandom(size - len(header))
    elif pattern == 'random_heavy':
        return os.urandom(int(size * 0.75)) + bytes(random.randint(65, 122) for _ in range(int(size * 0.25)))
    else:
        return os.urandom(size // 2) + bytes(random.randint(65, 122) for _ in range(size // 2))

def encrypt_aes(plaintext: bytes) -> bytes:
    key = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(plaintext, AES.block_size))

def encrypt_des(plaintext: bytes) -> bytes:
    key = os.urandom(8)
    cipher = DES.new(key, DES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(plaintext, DES.block_size))

_rsa_key = RSA.generate(2048)

def encrypt_rsa(plaintext: bytes) -> bytes:
    cipher = PKCS1_OAEP.new(_rsa_key)
    return cipher.encrypt(plaintext[:190])

def generate_dataset(output_path: str = "data/dataset.csv"):
    records = []
    for i in range(SAMPLES_PER_CLASS):
        plaintext = generate_realistic_plaintext(PLAINTEXT_SIZE)

        for label, ciphertext in [
            ("AES",       encrypt_aes(plaintext)),
            ("DES",       encrypt_des(plaintext)),
            ("RSA",       encrypt_rsa(plaintext)),
            ("plaintext", plaintext),
        ]:
            records.append({"bytes": list(ciphertext), "label": label})

        if i % 100 == 0:
            print(f"  Generated {i}/{SAMPLES_PER_CLASS} samples per class...")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved -> {output_path}  ({len(df)} total rows)")
    return df

if __name__ == "__main__":
    generate_dataset()
