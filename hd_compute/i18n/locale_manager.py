"""
Locale manager for handling internationalization and localization settings.
Provides locale detection, validation, and formatting utilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class LocaleType(Enum):
    """Types of locale formats supported."""
    LANGUAGE_ONLY = "language"      # e.g., "en", "fr"
    LANGUAGE_COUNTRY = "full"       # e.g., "en-US", "fr-CA"
    LANGUAGE_SCRIPT = "script"      # e.g., "zh-Hans", "zh-Hant"
    FULL_LOCALE = "complete"        # e.g., "zh-Hans-CN"

@dataclass
class LocaleInfo:
    """Information about a specific locale."""
    code: str
    language: str
    country: Optional[str]
    script: Optional[str]
    display_name: str
    native_name: str
    rtl: bool = False  # Right-to-left language
    plural_forms: int = 2  # Number of plural forms
    
class LocaleManager:
    """Manages locale settings and provides localization utilities."""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Supported locales with metadata
        self.supported_locales = {
            "en": LocaleInfo(
                code="en",
                language="English",
                country=None,
                script=None,
                display_name="English",
                native_name="English",
                rtl=False,
                plural_forms=2
            ),
            "en-US": LocaleInfo(
                code="en-US",
                language="English",
                country="United States",
                script=None,
                display_name="English (United States)",
                native_name="English (United States)",
                rtl=False,
                plural_forms=2
            ),
            "en-GB": LocaleInfo(
                code="en-GB",
                language="English",
                country="United Kingdom",
                script=None,
                display_name="English (United Kingdom)",
                native_name="English (United Kingdom)",
                rtl=False,
                plural_forms=2
            ),
            "es": LocaleInfo(
                code="es",
                language="Spanish",
                country=None,
                script=None,
                display_name="Spanish",
                native_name="Español",
                rtl=False,
                plural_forms=2
            ),
            "es-ES": LocaleInfo(
                code="es-ES",
                language="Spanish",
                country="Spain",
                script=None,
                display_name="Spanish (Spain)",
                native_name="Español (España)",
                rtl=False,
                plural_forms=2
            ),
            "es-MX": LocaleInfo(
                code="es-MX",
                language="Spanish",
                country="Mexico",
                script=None,
                display_name="Spanish (Mexico)",
                native_name="Español (México)",
                rtl=False,
                plural_forms=2
            ),
            "fr": LocaleInfo(
                code="fr",
                language="French",
                country=None,
                script=None,
                display_name="French",
                native_name="Français",
                rtl=False,
                plural_forms=2
            ),
            "fr-FR": LocaleInfo(
                code="fr-FR",
                language="French",
                country="France",
                script=None,
                display_name="French (France)",
                native_name="Français (France)",
                rtl=False,
                plural_forms=2
            ),
            "fr-CA": LocaleInfo(
                code="fr-CA",
                language="French",
                country="Canada",
                script=None,
                display_name="French (Canada)",
                native_name="Français (Canada)",
                rtl=False,
                plural_forms=2
            ),
            "de": LocaleInfo(
                code="de",
                language="German",
                country=None,
                script=None,
                display_name="German",
                native_name="Deutsch",
                rtl=False,
                plural_forms=2
            ),
            "de-DE": LocaleInfo(
                code="de-DE",
                language="German",
                country="Germany",
                script=None,
                display_name="German (Germany)",
                native_name="Deutsch (Deutschland)",
                rtl=False,
                plural_forms=2
            ),
            "ja": LocaleInfo(
                code="ja",
                language="Japanese",
                country=None,
                script=None,
                display_name="Japanese",
                native_name="日本語",
                rtl=False,
                plural_forms=1
            ),
            "ja-JP": LocaleInfo(
                code="ja-JP",
                language="Japanese",
                country="Japan",
                script=None,
                display_name="Japanese (Japan)",
                native_name="日本語 (日本)",
                rtl=False,
                plural_forms=1
            ),
            "zh": LocaleInfo(
                code="zh",
                language="Chinese",
                country=None,
                script=None,
                display_name="Chinese",
                native_name="中文",
                rtl=False,
                plural_forms=1
            ),
            "zh-CN": LocaleInfo(
                code="zh-CN",
                language="Chinese",
                country="China",
                script="Simplified",
                display_name="Chinese (Simplified, China)",
                native_name="中文 (简体, 中国)",
                rtl=False,
                plural_forms=1
            ),
            "zh-TW": LocaleInfo(
                code="zh-TW",
                language="Chinese",
                country="Taiwan",
                script="Traditional",
                display_name="Chinese (Traditional, Taiwan)",
                native_name="中文 (繁體, 台灣)",
                rtl=False,
                plural_forms=1
            ),
            "pt": LocaleInfo(
                code="pt",
                language="Portuguese",
                country=None,
                script=None,
                display_name="Portuguese",
                native_name="Português",
                rtl=False,
                plural_forms=2
            ),
            "pt-BR": LocaleInfo(
                code="pt-BR",
                language="Portuguese",
                country="Brazil",
                script=None,
                display_name="Portuguese (Brazil)",
                native_name="Português (Brasil)",
                rtl=False,
                plural_forms=2
            ),
            "ru": LocaleInfo(
                code="ru",
                language="Russian",
                country=None,
                script=None,
                display_name="Russian",
                native_name="Русский",
                rtl=False,
                plural_forms=3
            ),
            "ru-RU": LocaleInfo(
                code="ru-RU",
                language="Russian",
                country="Russia",
                script=None,
                display_name="Russian (Russia)",
                native_name="Русский (Россия)",
                rtl=False,
                plural_forms=3
            ),
            "ar": LocaleInfo(
                code="ar",
                language="Arabic",
                country=None,
                script=None,
                display_name="Arabic",
                native_name="العربية",
                rtl=True,
                plural_forms=6
            ),
            "ar-SA": LocaleInfo(
                code="ar-SA",
                language="Arabic",
                country="Saudi Arabia",
                script=None,
                display_name="Arabic (Saudi Arabia)",
                native_name="العربية (السعودية)",
                rtl=True,
                plural_forms=6
            )
        }
        
        # Locale patterns for validation
        self.locale_patterns = {
            LocaleType.LANGUAGE_ONLY: re.compile(r'^[a-z]{2}$'),
            LocaleType.LANGUAGE_COUNTRY: re.compile(r'^[a-z]{2}-[A-Z]{2}$'),
            LocaleType.LANGUAGE_SCRIPT: re.compile(r'^[a-z]{2}-[A-Z][a-z]{3}$'),
            LocaleType.FULL_LOCALE: re.compile(r'^[a-z]{2}-[A-Z][a-z]{3}-[A-Z]{2}$')
        }
        
        # Number formatting patterns
        self.number_formats = {
            "en": {"decimal": ".", "thousands": ",", "currency": "$"},
            "en-US": {"decimal": ".", "thousands": ",", "currency": "$"},
            "en-GB": {"decimal": ".", "thousands": ",", "currency": "£"},
            "es": {"decimal": ",", "thousands": ".", "currency": "€"},
            "es-MX": {"decimal": ".", "thousands": ",", "currency": "$"},
            "fr": {"decimal": ",", "thousands": " ", "currency": "€"},
            "de": {"decimal": ",", "thousands": ".", "currency": "€"},
            "ja": {"decimal": ".", "thousands": ",", "currency": "¥"},
            "zh": {"decimal": ".", "thousands": ",", "currency": "¥"},
            "zh-CN": {"decimal": ".", "thousands": ",", "currency": "¥"},
            "pt": {"decimal": ",", "thousands": ".", "currency": "€"},
            "pt-BR": {"decimal": ",", "thousands": ".", "currency": "R$"},
            "ru": {"decimal": ",", "thousands": " ", "currency": "₽"},
            "ar": {"decimal": ".", "thousands": ",", "currency": "ريال"}
        }
        
        # Date format patterns
        self.date_formats = {
            "en": "%m/%d/%Y",
            "en-US": "%m/%d/%Y",
            "en-GB": "%d/%m/%Y",
            "es": "%d/%m/%Y",
            "fr": "%d/%m/%Y",
            "de": "%d.%m.%Y",
            "ja": "%Y/%m/%d",
            "zh": "%Y/%m/%d",
            "zh-CN": "%Y/%m/%d",
            "pt": "%d/%m/%Y",
            "ru": "%d.%m.%Y",
            "ar": "%d/%m/%Y"
        }
        
        logger.info(f"Locale manager initialized with default locale: {default_locale}")
    
    def set_locale(self, locale: str) -> bool:
        """Set the current locale."""
        if not self.is_locale_supported(locale):
            # Try to find a fallback
            fallback = self.find_fallback_locale(locale)
            if fallback:
                logger.warning(f"Locale {locale} not supported, using fallback: {fallback}")
                locale = fallback
            else:
                logger.error(f"Locale {locale} not supported and no fallback found")
                return False
        
        self.current_locale = locale
        logger.info(f"Locale set to: {locale}")
        return True
    
    def get_current_locale(self) -> str:
        """Get the current locale."""
        return self.current_locale
    
    def is_locale_supported(self, locale: str) -> bool:
        """Check if a locale is supported."""
        return locale in self.supported_locales
    
    def validate_locale_format(self, locale: str) -> Optional[LocaleType]:
        """Validate locale format and return its type."""
        for locale_type, pattern in self.locale_patterns.items():
            if pattern.match(locale):
                return locale_type
        return None
    
    def parse_locale(self, locale: str) -> Dict[str, Optional[str]]:
        """Parse a locale string into components."""
        components = {
            "language": None,
            "script": None,
            "country": None
        }
        
        # Basic parsing - can be extended for more complex cases
        parts = locale.replace('_', '-').split('-')
        
        if len(parts) >= 1:
            components["language"] = parts[0].lower()
        
        if len(parts) >= 2:
            if len(parts[1]) == 2 and parts[1].isupper():
                components["country"] = parts[1]
            elif len(parts[1]) == 4 and parts[1][0].isupper():
                components["script"] = parts[1]
        
        if len(parts) >= 3:
            if len(parts[2]) == 2 and parts[2].isupper():
                components["country"] = parts[2]
        
        return components
    
    def find_fallback_locale(self, locale: str) -> Optional[str]:
        """Find a fallback locale for an unsupported locale."""
        components = self.parse_locale(locale)
        language = components.get("language")
        
        if not language:
            return self.default_locale
        
        # Try language-only fallback
        if language in self.supported_locales:
            return language
        
        # Try to find any locale with the same language
        for supported_locale in self.supported_locales:
            supported_components = self.parse_locale(supported_locale)
            if supported_components.get("language") == language:
                return supported_locale
        
        return self.default_locale
    
    def get_locale_info(self, locale: Optional[str] = None) -> Optional[LocaleInfo]:
        """Get information about a locale."""
        locale = locale or self.current_locale
        return self.supported_locales.get(locale)
    
    def get_supported_locales(self) -> List[str]:
        """Get list of all supported locale codes."""
        return list(self.supported_locales.keys())
    
    def get_locales_by_language(self, language: str) -> List[str]:
        """Get all supported locales for a specific language."""
        locales = []
        for locale_code, locale_info in self.supported_locales.items():
            if locale_code.startswith(language.lower()):
                locales.append(locale_code)
        return sorted(locales)
    
    def format_number(self, number: float, locale: Optional[str] = None, 
                     decimal_places: int = 2) -> str:
        """Format a number according to locale conventions."""
        locale = locale or self.current_locale
        format_info = self.number_formats.get(locale, self.number_formats["en"])
        
        # Format the number
        formatted = f"{number:,.{decimal_places}f}"
        
        # Replace separators according to locale
        if format_info["thousands"] != ",":
            parts = formatted.split(".")
            integer_part = parts[0].replace(",", format_info["thousands"])
            if len(parts) > 1:
                formatted = integer_part + format_info["decimal"] + parts[1]
            else:
                formatted = integer_part
        elif format_info["decimal"] != ".":
            formatted = formatted.replace(".", format_info["decimal"])
        
        return formatted
    
    def format_currency(self, amount: float, locale: Optional[str] = None,
                       decimal_places: int = 2) -> str:
        """Format a currency amount according to locale conventions."""
        locale = locale or self.current_locale
        format_info = self.number_formats.get(locale, self.number_formats["en"])
        
        formatted_number = self.format_number(amount, locale, decimal_places)
        currency_symbol = format_info["currency"]
        
        # Currency symbol position varies by locale
        if locale and locale.startswith(("ar",)):
            return f"{formatted_number} {currency_symbol}"
        else:
            return f"{currency_symbol}{formatted_number}"
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format a date according to locale conventions."""
        locale = locale or self.current_locale
        date_format = self.date_formats.get(locale, self.date_formats["en"])
        
        return date.strftime(date_format)
    
    def get_plural_form(self, count: int, locale: Optional[str] = None) -> int:
        """Get the plural form index for a count in the given locale."""
        locale = locale or self.current_locale
        locale_info = self.get_locale_info(locale)
        
        if not locale_info:
            return 0 if count == 1 else 1  # Default English pluralization
        
        # Simplified plural rules - real implementation would be more complex
        if locale_info.plural_forms == 1:
            return 0  # Languages like Chinese, Japanese
        elif locale_info.plural_forms == 2:
            return 0 if count == 1 else 1  # Most European languages
        elif locale_info.plural_forms == 3:
            # Russian-style pluralization (simplified)
            if count % 10 == 1 and count % 100 != 11:
                return 0
            elif 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
                return 1
            else:
                return 2
        elif locale_info.plural_forms == 6:
            # Arabic-style pluralization (very simplified)
            if count == 0:
                return 0
            elif count == 1:
                return 1
            elif count == 2:
                return 2
            elif 3 <= count <= 10:
                return 3
            elif 11 <= count <= 99:
                return 4
            else:
                return 5
        
        return 0 if count == 1 else 1
    
    def is_rtl_locale(self, locale: Optional[str] = None) -> bool:
        """Check if the locale uses right-to-left text direction."""
        locale = locale or self.current_locale
        locale_info = self.get_locale_info(locale)
        return locale_info.rtl if locale_info else False
    
    def detect_locale_from_accept_language(self, accept_language: str) -> str:
        """Detect best locale from HTTP Accept-Language header."""
        if not accept_language:
            return self.default_locale
        
        # Parse Accept-Language header
        languages = []
        for lang_entry in accept_language.split(','):
            lang_entry = lang_entry.strip()
            if ';' in lang_entry:
                lang, quality = lang_entry.split(';', 1)
                try:
                    quality = float(quality.split('=')[1])
                except (ValueError, IndexError):
                    quality = 1.0
            else:
                lang = lang_entry
                quality = 1.0
            
            languages.append((lang.strip(), quality))
        
        # Sort by quality
        languages.sort(key=lambda x: x[1], reverse=True)
        
        # Find best match
        for lang, _ in languages:
            # Try exact match first
            if lang in self.supported_locales:
                return lang
            
            # Try fallback
            fallback = self.find_fallback_locale(lang)
            if fallback and fallback != self.default_locale:
                return fallback
        
        return self.default_locale
    
    def get_locale_display_name(self, locale: str, display_locale: Optional[str] = None) -> str:
        """Get display name of a locale in another locale."""
        locale_info = self.get_locale_info(locale)
        if not locale_info:
            return locale
        
        display_locale = display_locale or self.current_locale
        
        # For now, return the display name in the locale's own language
        # In a full implementation, this would translate display names
        return locale_info.display_name

def main():
    """Example usage of LocaleManager."""
    manager = LocaleManager()
    
    # Test locale setting
    print(f"Current locale: {manager.get_current_locale()}")
    manager.set_locale("fr-FR")
    print(f"Locale changed to: {manager.get_current_locale()}")
    
    # Test number formatting
    number = 1234567.89
    print(f"Number in en-US: {manager.format_number(number, 'en-US')}")
    print(f"Number in fr-FR: {manager.format_number(number, 'fr-FR')}")
    print(f"Number in de-DE: {manager.format_number(number, 'de-DE')}")
    
    # Test currency formatting
    amount = 1234.56
    print(f"Currency in en-US: {manager.format_currency(amount, 'en-US')}")
    print(f"Currency in fr-FR: {manager.format_currency(amount, 'fr-FR')}")
    print(f"Currency in ja-JP: {manager.format_currency(amount, 'ja-JP')}")
    
    # Test date formatting
    date = datetime(2024, 3, 15)
    print(f"Date in en-US: {manager.format_date(date, 'en-US')}")
    print(f"Date in en-GB: {manager.format_date(date, 'en-GB')}")
    print(f"Date in de-DE: {manager.format_date(date, 'de-DE')}")
    
    # Test locale detection
    accept_header = "fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7, *;q=0.5"
    detected = manager.detect_locale_from_accept_language(accept_header)
    print(f"Detected locale from '{accept_header}': {detected}")
    
    return manager

if __name__ == "__main__":
    main()