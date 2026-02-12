"""Internationalization support for VisionAce."""

from i18n.en import EN
from i18n.ko import KO

_LANGUAGES = {
    "en": EN,
    "ko": KO,
}

_current_lang = "en"


def set_language(lang: str):
    global _current_lang
    if lang in _LANGUAGES:
        _current_lang = lang


def get_language() -> str:
    return _current_lang


def tr(key: str) -> str:
    """Translate a key to the current language string."""
    table = _LANGUAGES.get(_current_lang, EN)
    return table.get(key, key)
