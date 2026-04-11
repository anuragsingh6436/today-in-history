"""
Test configuration.

Sets dummy environment variables before any app module is imported
so that pydantic-settings validation passes during test collection.
"""

import os

# These must be set before importing anything from app.*
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
