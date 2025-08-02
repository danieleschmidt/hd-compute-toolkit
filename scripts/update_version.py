#!/usr/bin/env python3
"""Update version across all project files."""

import re
import sys
from pathlib import Path


def update_version(new_version: str) -> None:
    """Update version in all relevant files."""
    
    # Remove 'v' prefix if present
    version = new_version.lstrip('v')
    
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = re.sub(
            r'version = "[^"]*"',
            f'version = "{version}"',
            content
        )
        pyproject_path.write_text(content)
        print(f"✓ Updated pyproject.toml to version {version}")
    
    # Update __init__.py
    init_path = Path("hd_compute/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        
        # Update __version__ if it exists
        if '__version__' in content:
            content = re.sub(
                r'__version__ = "[^"]*"',
                f'__version__ = "{version}"',
                content
            )
        else:
            # Add __version__ if it doesn't exist
            content = f'"""HD-Compute-Toolkit: High-performance hyperdimensional computing."""\n\n__version__ = "{version}"\n\n' + content
        
        init_path.write_text(content)
        print(f"✓ Updated hd_compute/__init__.py to version {version}")
    
    # Update setup.cfg if it exists
    setup_cfg_path = Path("setup.cfg")
    if setup_cfg_path.exists():
        content = setup_cfg_path.read_text()
        content = re.sub(
            r'version = [^\n]*',
            f'version = {version}',
            content
        )
        setup_cfg_path.write_text(content)
        print(f"✓ Updated setup.cfg to version {version}")
    
    # Update docs/conf.py if it exists
    docs_conf_path = Path("docs/conf.py")
    if docs_conf_path.exists():
        content = docs_conf_path.read_text()
        
        # Update version and release
        content = re.sub(
            r"version = '[^']*'",
            f"version = '{version}'",
            content
        )
        content = re.sub(
            r"release = '[^']*'",
            f"release = '{version}'",
            content
        )
        
        docs_conf_path.write_text(content)
        print(f"✓ Updated docs/conf.py to version {version}")
    
    print(f"🎉 Successfully updated all files to version {version}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format (basic semver check)
    version_pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?(?:\+[a-zA-Z0-9.-]+)?$'
    if not re.match(version_pattern, new_version.lstrip('v')):
        print(f"Error: Invalid version format: {new_version}")
        print("Expected format: X.Y.Z or X.Y.Z-prerelease or X.Y.Z+build")
        sys.exit(1)
    
    update_version(new_version)


if __name__ == "__main__":
    main()