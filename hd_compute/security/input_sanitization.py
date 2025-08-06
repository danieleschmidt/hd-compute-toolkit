"""Input sanitization and validation utilities."""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union
import urllib.parse

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Input sanitization utilities for secure data handling."""
    
    def __init__(self):
        self.malicious_patterns = [
            # SQL Injection patterns
            r"(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(--\s|\/\*|\*\/|;)",
            
            # Code injection patterns  
            r"(__import__|eval\s*\(|exec\s*\()",
            r"(system\s*\(|popen\s*\(|subprocess)",
            
            # Path traversal patterns
            r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
            
            # Script injection patterns
            r"(<script|javascript:|vbscript:|onload=|onerror=)",
            
            # Command injection patterns
            r"(;\s*(?:rm|del|format|shutdown|reboot)|\|\s*(?:rm|del|format))",
        ]
        
        self.allowed_filename_chars = re.compile(r'^[a-zA-Z0-9._-]+$')
        self.allowed_path_chars = re.compile(r'^[a-zA-Z0-9._/-]+$')
    
    def sanitize_string(self, input_string: str, max_length: int = 1000) -> str:
        """Sanitize general string input.
        
        Args:
            input_string: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_string, str):
            input_string = str(input_string)
        
        # Truncate if too long
        if len(input_string) > max_length:
            input_string = input_string[:max_length]
            logger.warning(f"Input truncated to {max_length} characters")
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in input_string 
                          if ord(char) >= 32 or char in ['\n', '\r', '\t'])
        
        # HTML encode potentially dangerous characters
        sanitized = html.escape(sanitized)
        
        return sanitized
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for security.
        
        Args:
            filename: Filename to validate
            
        Returns:
            True if filename is safe
        """
        if not filename or len(filename) == 0:
            return False
        
        # Check length
        if len(filename) > 255:
            logger.warning(f"Filename too long: {len(filename)} characters")
            return False
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\\\' in filename:
            logger.warning(f"Path traversal detected in filename: {filename}")
            return False
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        
        if filename.upper().split('.')[0] in reserved_names:
            logger.warning(f"Reserved filename detected: {filename}")
            return False
        
        # Check allowed characters
        if not self.allowed_filename_chars.match(filename):
            logger.warning(f"Invalid characters in filename: {filename}")
            return False
        
        return True
    
    def sanitize_path(self, path: str) -> Optional[str]:
        """Sanitize file path.
        
        Args:
            path: File path to sanitize
            
        Returns:
            Sanitized path or None if invalid
        """
        if not path:
            return None
        
        # Normalize path separators
        path = path.replace('\\\\', '/').replace('\\', '/')
        
        # Remove duplicate slashes
        path = re.sub(r'/+', '/', path)
        
        # Check for path traversal
        if '..' in path:
            logger.warning(f"Path traversal detected: {path}")
            return None
        
        # Check for absolute paths (if not allowed)
        if path.startswith('/'):
            logger.warning(f"Absolute path detected: {path}")
            return None
        
        # Validate characters
        if not self.allowed_path_chars.match(path):
            logger.warning(f"Invalid characters in path: {path}")
            return None
        
        return path
    
    def validate_url(self, url: str) -> bool:
        """Validate URL for security.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is safe
        """
        if not url:
            return False
        
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                logger.warning(f"Invalid URL scheme: {parsed.scheme}")
                return False
            
            # Check for localhost/private networks (if not allowed)
            hostname = parsed.hostname
            if hostname:
                if hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                    logger.warning(f"Localhost URL detected: {hostname}")
                    return False
                
                # Check for private IP ranges
                if self._is_private_ip(hostname):
                    logger.warning(f"Private IP detected: {hostname}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"URL validation error: {e}")
            return False
    
    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname is a private IP address."""
        try:
            import ipaddress
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except:
            return False
    
    def sanitize_sql_input(self, input_value: str) -> str:
        """Sanitize input for SQL queries.
        
        Args:
            input_value: Input to sanitize
            
        Returns:
            Sanitized input
        """
        if not isinstance(input_value, str):
            input_value = str(input_value)
        
        # Remove SQL injection patterns
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', '\\']
        for char in dangerous_chars:
            input_value = input_value.replace(char, '')
        
        # Remove SQL keywords (case insensitive)
        sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', 'SELECT', 
                       'CREATE', 'ALTER', 'EXEC', 'EXECUTE']
        
        for keyword in sql_keywords:
            input_value = re.sub(rf'\\b{keyword}\\b', '', input_value, flags=re.IGNORECASE)
        
        return input_value.strip()
    
    def detect_malicious_input(self, input_data: str) -> List[str]:
        """Detect potentially malicious input patterns.
        
        Args:
            input_data: Input to analyze
            
        Returns:
            List of detected malicious patterns
        """
        detected_patterns = []
        
        for pattern in self.malicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                detected_patterns.append(pattern)
                logger.warning(f"Malicious pattern detected: {pattern}")
        
        return detected_patterns
    
    def sanitize_json_input(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON input recursively.
        
        Args:
            json_data: JSON data to sanitize
            
        Returns:
            Sanitized JSON data
        """
        if isinstance(json_data, dict):
            return {
                self.sanitize_string(str(key), 100): self.sanitize_json_input(value)
                for key, value in json_data.items()
            }
        elif isinstance(json_data, list):
            return [self.sanitize_json_input(item) for item in json_data]
        elif isinstance(json_data, str):
            return self.sanitize_string(json_data)
        else:
            return json_data
    
    def validate_hypervector_data(self, hv_data: Any) -> bool:
        """Validate hypervector data for security.
        
        Args:
            hv_data: Hypervector data to validate
            
        Returns:
            True if data is safe
        """
        # Check data type
        if hasattr(hv_data, 'data') and hasattr(hv_data, 'shape'):
            # SimpleArray-like object
            data_list = hv_data.data
        elif isinstance(hv_data, (list, tuple)):
            data_list = hv_data
        else:
            logger.warning(f"Unexpected hypervector data type: {type(hv_data)}")
            return False
        
        # Check data size (prevent memory exhaustion)
        max_size = 1000000  # 1M elements max
        if len(data_list) > max_size:
            logger.warning(f"Hypervector data too large: {len(data_list)} elements")
            return False
        
        # Check data values (should be numeric)
        try:
            for value in data_list[:100]:  # Sample first 100 values
                if not isinstance(value, (int, float)) or abs(value) > 1e6:
                    logger.warning(f"Invalid hypervector value: {value}")
                    return False
        except Exception as e:
            logger.warning(f"Error validating hypervector data: {e}")
            return False
        
        return True
    
    def sanitize_config_value(self, key: str, value: Any) -> Any:
        """Sanitize configuration values.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            Sanitized value
        """
        # Sensitive keys that should be strings
        if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
            if isinstance(value, str):
                # Don't log sensitive values
                return self.sanitize_string(value, 255)
            else:
                logger.warning(f"Sensitive config key {key} has non-string value")
                return str(value)
        
        # Path values
        if 'path' in key.lower() or 'dir' in key.lower() or 'file' in key.lower():
            if isinstance(value, str):
                sanitized_path = self.sanitize_path(value)
                if sanitized_path is None:
                    logger.warning(f"Invalid path in config: {key}={value}")
                    return ""
                return sanitized_path
        
        # URL values
        if 'url' in key.lower() or 'endpoint' in key.lower():
            if isinstance(value, str):
                if not self.validate_url(value):
                    logger.warning(f"Invalid URL in config: {key}={value}")
                    return ""
                return value
        
        # Numeric values
        if isinstance(value, (int, float)):
            # Prevent extremely large values
            if abs(value) > 1e10:
                logger.warning(f"Extremely large numeric value in config: {key}={value}")
                return 0
            return value
        
        # String values
        if isinstance(value, str):
            return self.sanitize_string(value, 1000)
        
        # Other types (bool, list, dict)
        return value
    
    def create_security_headers(self) -> Dict[str, str]:
        """Create security headers for HTTP responses.
        
        Returns:
            Dictionary of security headers
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }