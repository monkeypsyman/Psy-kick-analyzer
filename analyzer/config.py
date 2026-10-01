"""Cross-cutting constants. Things that don't belong to any one stage."""

SCHEMA_VERSION = 1
KIND = 'kick'

# f0 consensus: agreement above this flags an f-ambiguous sample.
F_AGREE_FLAG_HZ = 5.0

# Multi-method f0 search band defaults.
F0_LO_HZ = 25.0
F0_HI_HZ = 250.0

# Class discriminator: raw_peak_ms at or below this -> fishtail.
# Above -> swell. See PROJECT_STATE §3 (S5 finding).
FISHTAIL_PEAK_MAX_MS = 8.0

# Content-ID: how much of the head to hash, and hash length.
ID_SECONDS = 5.0
ID_HASH_LEN = 12