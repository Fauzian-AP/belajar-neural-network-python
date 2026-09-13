from pydantic import ValidationError
from src.core.activation_functions import LeakyReLU  # Sesuaikan path import file kamu

# --- TEST 1: Nilai Valid (Harus Berhasil) ---
try:
    layer_valid = LeakyReLU(alpha=0.05)
    print("✅ Test Valid Berhasil! Nilai:", layer_valid.alpha)
except ValidationError as e:
    print("❌ Error tidak terduga pada nilai valid:", e)


# --- TEST 2: Nilai Invalid / Di luar rentang (Harus Error) ---
try:
    layer_invalid = LeakyReLU(alpha=1.5)  # Melanggar lt=1.0
    print("❌ GAGAL: Pydantic meloloskan nilai 1.5 padahal harusnya diblokir!")
except ValidationError as e:
    print("✅ Test Invalid Berhasil! Pydantic sukses memblokir 1.5:")
    print(e)