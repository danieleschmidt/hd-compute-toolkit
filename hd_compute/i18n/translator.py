"""Translation utilities for multi-language support."""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Translator:
    """Multi-language translation manager."""
    
    def __init__(self, locale: str = 'en', translations_dir: str = None):
        """Initialize translator.
        
        Args:
            locale: Default locale (language code)
            translations_dir: Directory containing translation files
        """
        self.locale = locale
        self.translations_dir = Path(translations_dir) if translations_dir else Path(__file__).parent / 'translations'
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_locale = 'en'
        
        # Supported locales
        self.supported_locales = [
            'en',  # English
            'es',  # Spanish
            'fr',  # French
            'de',  # German
            'ja',  # Japanese
            'zh',  # Chinese (Simplified)
            'pt',  # Portuguese
            'ru',  # Russian
        ]
        
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        if not self.translations_dir.exists():
            logger.warning(f"Translations directory not found: {self.translations_dir}")
            self._create_default_translations()
            return
        
        for locale in self.supported_locales:
            translation_file = self.translations_dir / f"{locale}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[locale] = json.load(f)
                    logger.debug(f"Loaded translations for {locale}")
                except Exception as e:
                    logger.error(f"Error loading translations for {locale}: {e}")
            else:
                logger.warning(f"Translation file not found for {locale}: {translation_file}")
        
        # Create default English translations if none exist
        if 'en' not in self.translations:
            self._create_default_translations()
    
    def _create_default_translations(self):
        """Create default English translations."""
        default_translations = {
            # General terms
            "welcome": "Welcome",
            "error": "Error",
            "warning": "Warning",
            "success": "Success",
            "loading": "Loading",
            "complete": "Complete",
            "failed": "Failed",
            
            # HDC-specific terms
            "hypervector": "Hypervector",
            "dimension": "Dimension",
            "sparsity": "Sparsity",
            "similarity": "Similarity",
            "bundle_operation": "Bundle Operation",
            "bind_operation": "Bind Operation",
            "cleanup_operation": "Cleanup Operation",
            
            # Operations
            "generating_hypervector": "Generating hypervector",
            "bundling_hypervectors": "Bundling hypervectors",
            "binding_hypervectors": "Binding hypervectors",
            "calculating_similarity": "Calculating similarity",
            "encoding_sequence": "Encoding sequence",
            
            # Memory operations
            "item_memory": "Item Memory",
            "associative_memory": "Associative Memory",
            "storing_pattern": "Storing pattern",
            "recalling_pattern": "Recalling pattern",
            "memory_cleanup": "Memory cleanup",
            
            # Performance and benchmarking
            "benchmark": "Benchmark",
            "performance": "Performance",
            "profiling": "Profiling",
            "optimization": "Optimization",
            "execution_time": "Execution time",
            "operations_per_second": "Operations per second",
            "memory_usage": "Memory usage",
            
            # Errors and validation
            "invalid_dimension": "Invalid dimension",
            "invalid_sparsity": "Invalid sparsity level",
            "dimension_mismatch": "Dimension mismatch",
            "empty_input": "Empty input not allowed",
            "validation_failed": "Validation failed",
            "parameter_error": "Parameter error",
            
            # Security
            "security_scan": "Security scan",
            "vulnerability_detected": "Vulnerability detected",
            "audit_log": "Audit log",
            "authentication_failed": "Authentication failed",
            "access_denied": "Access denied",
            
            # Configuration
            "configuration": "Configuration",
            "settings": "Settings",
            "environment": "Environment",
            "device": "Device",
            "backend": "Backend",
            
            # File operations
            "file_not_found": "File not found",
            "invalid_file_format": "Invalid file format",
            "export_successful": "Export successful",
            "import_successful": "Import successful",
            
            # API and CLI
            "api_request": "API request",
            "command_executed": "Command executed",
            "help_message": "Help message",
            "usage": "Usage",
            "options": "Options",
        }
        
        self.translations['en'] = default_translations
        
        # Save default translations
        self.translations_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.translations_dir / 'en.json', 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
            logger.info("Created default English translations")
        except Exception as e:
            logger.error(f"Error saving default translations: {e}")
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the specified locale.
        
        Args:
            key: Translation key
            locale: Target locale (uses instance default if None)
            **kwargs: Variables for string formatting
            
        Returns:
            Translated string
        """
        target_locale = locale or self.locale
        
        # Get translation
        translation = self._get_translation(key, target_locale)
        
        # Apply string formatting if variables provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error formatting translation '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, locale: str) -> str:
        """Get translation for key in specified locale."""
        # Try target locale
        if locale in self.translations and key in self.translations[locale]:
            return self.translations[locale][key]
        
        # Try fallback locale
        if (locale != self.fallback_locale and 
            self.fallback_locale in self.translations and 
            key in self.translations[self.fallback_locale]):
            logger.debug(f"Using fallback translation for '{key}' ({locale} -> {self.fallback_locale})")
            return self.translations[self.fallback_locale][key]
        
        # Return key if no translation found
        logger.warning(f"No translation found for '{key}' in locale '{locale}'")
        return key
    
    def set_locale(self, locale: str):
        """Set current locale.
        
        Args:
            locale: New locale to use
        """
        if locale not in self.supported_locales:
            logger.warning(f"Unsupported locale: {locale}. Using default: {self.locale}")
            return
        
        self.locale = locale
        logger.info(f"Locale changed to: {locale}")
    
    def get_available_locales(self) -> list:
        """Get list of available locales with translations."""
        return list(self.translations.keys())
    
    def add_translations(self, locale: str, translations: Dict[str, str]):
        """Add translations for a locale.
        
        Args:
            locale: Locale code
            translations: Dictionary of key-value translations
        """
        if locale not in self.translations:
            self.translations[locale] = {}
        
        self.translations[locale].update(translations)
        logger.info(f"Added {len(translations)} translations for {locale}")
    
    def export_translations(self, locale: str, filename: str):
        """Export translations to file.
        
        Args:
            locale: Locale to export
            filename: Output filename
        """
        if locale not in self.translations:
            logger.error(f"No translations found for locale: {locale}")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.translations[locale], f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {locale} translations to {filename}")
        except Exception as e:
            logger.error(f"Error exporting translations: {e}")
    
    def import_translations(self, locale: str, filename: str):
        """Import translations from file.
        
        Args:
            locale: Target locale
            filename: Input filename
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            self.add_translations(locale, translations)
            logger.info(f"Imported translations for {locale} from {filename}")
            
        except Exception as e:
            logger.error(f"Error importing translations: {e}")
    
    def get_translation_coverage(self, locale: str) -> float:
        """Get translation coverage percentage for a locale.
        
        Args:
            locale: Locale to check
            
        Returns:
            Coverage percentage (0-100)
        """
        if locale not in self.translations or self.fallback_locale not in self.translations:
            return 0.0
        
        base_keys = set(self.translations[self.fallback_locale].keys())
        locale_keys = set(self.translations[locale].keys())
        
        if not base_keys:
            return 100.0
        
        coverage = len(locale_keys & base_keys) / len(base_keys) * 100
        return coverage
    
    def find_missing_translations(self, locale: str) -> list:
        """Find missing translation keys for a locale.
        
        Args:
            locale: Locale to check
            
        Returns:
            List of missing keys
        """
        if locale not in self.translations or self.fallback_locale not in self.translations:
            return []
        
        base_keys = set(self.translations[self.fallback_locale].keys())
        locale_keys = set(self.translations[locale].keys())
        
        missing_keys = list(base_keys - locale_keys)
        return missing_keys
    
    # Convenience methods for common translations
    def error_message(self, message: str, **kwargs) -> str:
        """Get localized error message."""
        return f"{self.translate('error')}: {self.translate(message, **kwargs)}"
    
    def success_message(self, message: str, **kwargs) -> str:
        """Get localized success message."""
        return f"{self.translate('success')}: {self.translate(message, **kwargs)}"
    
    def warning_message(self, message: str, **kwargs) -> str:
        """Get localized warning message."""
        return f"{self.translate('warning')}: {self.translate(message, **kwargs)}"


# Global translator instance
_global_translator: Optional[Translator] = None


def get_translator() -> Translator:
    """Get global translator instance."""
    global _global_translator
    if _global_translator is None:
        _global_translator = Translator()
    return _global_translator


def t(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translations.
    
    Args:
        key: Translation key
        locale: Target locale (optional)
        **kwargs: Variables for string formatting
        
    Returns:
        Translated string
    """
    return get_translator().translate(key, locale, **kwargs)


def set_global_locale(locale: str):
    """Set global locale for all translations."""
    get_translator().set_locale(locale)