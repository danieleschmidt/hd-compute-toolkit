# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

1. **Do not open a public issue** for security vulnerabilities
2. Email security details to: security@example.com
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce the issue
5. If possible, include a proof of concept

## What to Report

Please report any security issues including:

- Authentication bypasses
- Data exposure vulnerabilities  
- Injection attacks (code injection, etc.)
- Memory corruption issues in FPGA/Vulkan kernels
- Dependency vulnerabilities with active exploits

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days  
- **Security Patch**: Within 30 days (for confirmed issues)

## Security Best Practices

When using HD-Compute-Toolkit:

- Keep dependencies updated
- Validate input data before processing
- Use secure coding practices in custom kernels
- Monitor for security advisories
- Run security scans in CI/CD pipelines

## Recognition

We acknowledge security researchers who responsibly disclose vulnerabilities. With your permission, we will:

- Credit you in our security advisories
- Include you in our contributors list
- Provide a reference for responsible disclosure

Thank you for helping keep HD-Compute-Toolkit secure!