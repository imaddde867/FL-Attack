"""
Homomorphic Encryption for Federated Learning.

Implements a minimal Paillier-like additive HE scheme for secure gradient aggregation.
"""

import secrets
import math
from typing import Tuple, List


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _invmod(a: int, m: int) -> int:
    """Modular inverse of a mod m."""
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m


def _is_probable_prime(n: int, k: int = 8) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p

    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    """Generate a random probable prime of specified bit length."""
    while True:
        p = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(p):
            return p


class HomomorphicEncryptor:
    """
    Minimal Paillier-like additive homomorphic encryption.
    
    Supports:
        - Encryption/decryption of integers and floats
        - Homomorphic addition of ciphertexts
        - Scalar multiplication of ciphertexts
        - Encrypted vector aggregation for FL
    
    Note: Floats are scaled by `precision` for fixed-point representation.
    """

    def __init__(self, bits: int = 512, precision: int = 1_000_000):
        """
        Initialize HE with new key pair.
        
        Args:
            bits: Key size in bits (security parameter)
            precision: Scaling factor for float-to-int conversion
        """
        self.bits = int(bits)
        self.precision = int(precision)
        self._generate_keys()

    def _generate_keys(self) -> None:
        """Generate Paillier key pair."""
        half = self.bits // 2
        self.p = _generate_prime(half)
        self.q = self.p
        while self.q == self.p:
            self.q = _generate_prime(half)

        self.n = self.p * self.q
        self.n_sq = self.n * self.n
        self.g = self.n + 1

        # λ = lcm(p-1, q-1)
        lam = (self.p - 1) * (self.q - 1) // math.gcd(self.p - 1, self.q - 1)
        self._lambda = lam
        self.mu = _invmod(lam % self.n, self.n)

    def public_key(self) -> Tuple[int, int, int]:
        """Return public key components (n, n², g)."""
        return (self.n, self.n_sq, self.g)

    def encrypt_int(self, m: int) -> int:
        """Encrypt an integer message."""
        m = m % self.n
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if math.gcd(r, self.n) == 1:
                break
        return (pow(self.g, m, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq

    def decrypt_int(self, c: int) -> int:
        """Decrypt a ciphertext to integer (signed in [-n/2, n/2])."""
        x = pow(c, self._lambda, self.n_sq)
        m = ((x - 1) // self.n * self.mu) % self.n
        return m - self.n if m > self.n // 2 else m

    def add_cipher(self, c1: int, c2: int) -> int:
        """Homomorphic addition: E(m1) × E(m2) = E(m1 + m2)."""
        return (c1 * c2) % self.n_sq

    def mul_const(self, c: int, k: int) -> int:
        """Scalar multiplication: E(m)^k = E(k × m)."""
        return pow(c, k, self.n_sq)

    def _to_int(self, value: float) -> int:
        """Convert float to fixed-point integer."""
        return int(round(value * self.precision))

    def _to_float(self, m: int) -> float:
        """Convert fixed-point integer back to float."""
        return float(m) / self.precision

    def encrypt_vector(self, vals: List[float]) -> List[int]:
        """Encrypt a list of floats."""
        return [self.encrypt_int(self._to_int(v)) for v in vals]

    def decrypt_vector(self, cts: List[int]) -> List[float]:
        """Decrypt a list of ciphertexts to floats."""
        return [self._to_float(self.decrypt_int(c)) for c in cts]

    def _sample_laplace(self, scale: float) -> int:
        """Sample discretized Laplace noise."""
        u = secrets.randbelow(10**9) / 10**9
        sign = 1 if u >= 0.5 else -1
        v = -math.log(1 - 2 * abs(u - 0.5))
        return int(round(sign * v * scale * self.precision))

    def add_noise_plain(self, vals: List[float], scale: float) -> List[float]:
        """Add Laplace noise to plaintext values."""
        return [v + self._sample_laplace(scale) / self.precision for v in vals]

    def add_noise_encrypted(self, cts: List[int], scale: float) -> List[int]:
        """Add encrypted Laplace noise homomorphically."""
        return [self.add_cipher(c, self.encrypt_int(self._sample_laplace(scale))) for c in cts]

    def aggregate_encrypted_vectors(self, list_of_cts: List[List[int]]) -> List[int]:
        """
        Aggregate multiple encrypted vectors via homomorphic addition.
        
        Args:
            list_of_cts: List of encrypted vectors (same length)
        
        Returns:
            Elementwise encrypted sum
        """
        if not list_of_cts:
            return []

        length = len(list_of_cts[0])
        result = [1] * length  # E(0) neutral element

        for cts in list_of_cts:
            if len(cts) != length:
                raise ValueError("All ciphertext vectors must have same length")
            for i in range(length):
                result[i] = (result[i] * cts[i]) % self.n_sq

        return result