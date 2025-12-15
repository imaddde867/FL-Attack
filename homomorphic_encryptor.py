import secrets
import math
from typing import Tuple, List


# basic number theory helpers

def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    # extended Euclidean algorithm
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _invmod(a: int, m: int) -> int:
    # modular inverse of a mod m
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m


def _is_probable_prime(n: int, k: int = 8) -> bool:
    # Miller–Rabin primality test
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def _generate_prime(bits: int) -> int:
    # random odd with top bit set, loop until probable prime
    while True:
        p = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(p):
            return p


class HomomorphicEncryptor:
    """
    Minimal Paillier-like additive HE.

    Public key: n = p*q, g = n + 1
    Secret key: lambda, mu
    E(m) = g^m * r^n mod n^2
    D(c) = L(c^lambda mod n^2) * mu mod n,  L(x) = (x - 1)//n

    Integers only; floats are scaled by `precision`.
    """

    def __init__(self, bits: int = 512, precision: int = 1_000_000):
        self.bits = int(bits)
        self.precision = int(precision)
        self._generate_keys()

    def _generate_keys(self) -> None:
        # generate two distinct primes of size ~bits/2
        half = self.bits // 2

        p = _generate_prime(half)
        q = p
        while q == p:
            q = _generate_prime(half)

        self.p = p
        self.q = q
        self.n = p * q
        self.n_sq = self.n * self.n
        self.g = self.n + 1

        # lambda = lcm(p-1, q-1)
        lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
        self._lambda = lam

        lam_mod_n = lam % self.n
        # for g = n+1, L(g^lambda mod n^2) = lambda mod n
        self.mu = _invmod(lam_mod_n, self.n)

    def public_key(self) -> Tuple[int, int, int]:
        return (self.n, self.n_sq, self.g)

    # integer encrypt / decrypt

    def encrypt_int(self, m: int) -> int:
        # map into Z_n
        m = m % self.n

        # sample r coprime with n
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if math.gcd(r, self.n) == 1:
                break

        return (pow(self.g, m, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq

    def decrypt_int(self, c: int) -> int:
        x = pow(c, self._lambda, self.n_sq)
        Lx = (x - 1) // self.n
        m = (Lx * self.mu) % self.n

        # interpret as signed integer in (-n/2, n/2]
        if m > self.n // 2:
            m -= self.n

        return m

    # homomorphic ops on ciphertexts

    def add_cipher(self, c1: int, c2: int) -> int:
        # E(m1) * E(m2) = E(m1 + m2)
        return (c1 * c2) % self.n_sq

    def mul_const(self, c: int, k: int) -> int:
        # E(m)^k = E(k*m)
        return pow(c, k, self.n_sq)

    # float helpers (for model params)

    def _to_int(self, value: float) -> int:
        return int(round(value * self.precision))

    def _to_float(self, m: int) -> float:
        return float(m) / self.precision

    def encrypt_vector(self, vals: List[float]) -> List[int]:
        # encrypt list of floats as ciphertexts
        return [self.encrypt_int(self._to_int(v)) for v in vals]

    def decrypt_vector(self, cts: List[int]) -> List[float]:
        # decrypt list of ciphertexts back to floats
        return [self._to_float(self.decrypt_int(c)) for c in cts]

    # simple Laplace noise (optional DP)

    def _sample_laplace(self, scale: float) -> int:
        # Laplace(0, scale) using inverse CDF, discretized by precision
        u = secrets.randbelow(10**9) / 10**9
        sign = 1 if u >= 0.5 else -1
        v = -math.log(1 - 2 * abs(u - 0.5))
        noise = sign * v * scale
        return int(round(noise * self.precision))

    def add_noise_plain(self, vals: List[float], scale: float) -> List[float]:
        return [v + (self._sample_laplace(scale) / self.precision) for v in vals]

    def add_noise_encrypted(self, cts: List[int], scale: float) -> List[int]:
        # encrypt Laplace noise and add homomorphically
        noisy = []
        for c in cts:
            noise_int = self._sample_laplace(scale)
            enc_noise = self.encrypt_int(noise_int)
            noisy.append(self.add_cipher(c, enc_noise))
        return noisy

    # encrypted vector aggregation (for FL)

    def aggregate_encrypted_vectors(self, list_of_cts: List[List[int]]) -> List[int]:
        # elementwise homomorphic sum: out[i] = product_j cts_j[i] mod n^2
        if not list_of_cts:
            return []

        length = len(list_of_cts[0])
        out = [1] * length  # E(0) neutral element

        for cts in list_of_cts:
            if len(cts) != length:
                raise ValueError("All ciphertext vectors must have same length")
            for i in range(length):
                out[i] = (out[i] * cts[i]) % self.n_sq

        return out

"""Usage example (uncomment to run)"""
# if __name__ == "__main__":
   # enc = HomomorphicEncryptor(bits=256, precision=1000)

    # 1) encrypt/decrypt round‑trip
   # vals = [0.5, -1.25, 3.0]
   # cts = enc.encrypt_vector(vals)
   # dec = enc.decrypt_vector(cts)
   # print("HE orig:", vals)
   # print("HE dec :", dec)

    # 2) homomorphic sum of two vectors
    # vals2 = [1.0, 2.0, -4.0]
    # cts2 = enc.encrypt_vector(vals2)
    # agg = enc.aggregate_encrypted_vectors([cts, cts2])
    # dec_agg = enc.decrypt_vector(agg)
    # print("sum orig:", [a + b for a, b in zip(vals, vals2)])
    # print("sum dec :", dec_agg)

"""
output:
HE orig: [0.5, -1.25, 3.0]
HE dec : [0.5, -1.25, 3.0]
sum orig: [1.5, 0.75, -1.0]
sum dec : [1.5, 0.75, -1.0]

"""