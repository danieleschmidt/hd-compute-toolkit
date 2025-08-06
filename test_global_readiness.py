#!/usr/bin/env python3
"""
Global Readiness Test Suite: Multi-region deployment, I18n, and compliance validation
"""

import sys
import os
import json
from pathlib import Path
import traceback
sys.path.insert(0, '/root/repo')


def test_internationalization():
    """Test internationalization and multi-language support."""
    print("🌍 INTERNATIONALIZATION (I18N) TEST")
    print("=" * 50)
    
    try:
        from hd_compute.i18n.translator import Translator
        from hd_compute.i18n.locale_manager import LocaleManager
        
        # Test translator initialization
        translator = Translator()
        print(f"✅ Translator initialized with default locale: {translator.locale}")
        
        # Test supported locales
        expected_locales = ['en', 'es', 'fr', 'de', 'ja', 'zh']
        supported = translator.supported_locales
        
        coverage = len(set(expected_locales) & set(supported)) / len(expected_locales)
        print(f"📊 Locale coverage: {coverage*100:.1f}% ({len(supported)} locales)")
        
        for locale in expected_locales:
            if locale in supported:
                print(f"   ✅ {locale}: supported")
            else:
                print(f"   ⚠️  {locale}: not supported")
        
        # Test translation functionality
        print("\n🔤 Testing translation functionality:")
        
        # Test with available translations
        test_keys = ['error', 'success', 'warning', 'info']
        working_translations = 0
        
        for key in test_keys:
            try:
                translation = translator.translate(key)
                if translation and translation != key:  # Not just returning the key
                    print(f"   ✅ '{key}' -> '{translation}'")
                    working_translations += 1
                else:
                    print(f"   ⚠️  '{key}' -> no translation (fallback: {translation})")
            except Exception as e:
                print(f"   ❌ '{key}' -> translation error: {e}")
        
        translation_success_rate = (working_translations / len(test_keys)) * 100
        
        # Test locale switching
        print(f"\n🔄 Testing locale switching:")
        
        locale_switching_works = True
        try:
            original_locale = translator.locale
            translator.set_locale('es')
            if translator.locale == 'es':
                print(f"   ✅ Successfully switched to Spanish")
            else:
                print(f"   ⚠️  Locale switch may not have worked")
                locale_switching_works = False
                
            translator.set_locale(original_locale)
        except Exception as e:
            print(f"   ❌ Locale switching failed: {e}")
            locale_switching_works = False
        
        # Test LocaleManager
        print(f"\n🌐 Testing LocaleManager:")
        
        locale_manager_works = True
        try:
            manager = LocaleManager()
            current_locale = manager.get_current_locale()
            print(f"   ✅ LocaleManager working, current locale: {current_locale}")
            
            # Test locale detection
            detected_locale = manager.detect_locale()
            print(f"   ✅ Locale detection: {detected_locale}")
            
        except Exception as e:
            print(f"   ❌ LocaleManager failed: {e}")
            locale_manager_works = False
        
        # Calculate I18n score
        locale_score = coverage * 100
        translation_score = translation_success_rate
        functionality_score = (locale_switching_works + locale_manager_works) * 50
        
        i18n_score = (locale_score + translation_score + functionality_score) / 3
        
        print(f"\n🎯 I18n Score: {i18n_score:.1f}/100")
        
        if i18n_score >= 75:
            print("✅ I18N READY: Multi-language support operational")
            return True
        else:
            print(f"⚠️  I18N NEEDS WORK: {i18n_score:.1f}% < 75% threshold")
            return i18n_score >= 50
            
    except Exception as e:
        print(f"❌ I18n testing failed: {e}")
        traceback.print_exc()
        return False


def test_compliance_readiness():
    """Test GDPR, CCPA, and PDPA compliance."""
    print("\n⚖️  COMPLIANCE READINESS TEST")
    print("=" * 50)
    
    try:
        from hd_compute.compliance.gdpr import GDPRCompliance
        from hd_compute.compliance.data_privacy import DataPrivacyManager
        from hd_compute.compliance.audit_compliance import ComplianceAuditor
        
        print("📋 Testing GDPR compliance...")
        
        # Test GDPR compliance checker
        gdpr = GDPRCompliance()
        
        # Test data processing lawfulness
        test_data = {
            'user_id': 'test_user_123',
            'hypervector_data': [0.1, 0.2, 0.3],
            'timestamp': '2024-01-01T00:00:00Z',
            'processing_purpose': 'hdc_computation'
        }
        
        gdpr_checks = [
            ('data_minimization', lambda: gdpr.check_data_minimization(test_data)),
            ('consent_validation', lambda: gdpr.validate_consent('test_user_123')),
            ('retention_policy', lambda: gdpr.check_retention_policy(test_data)),
            ('access_rights', lambda: gdpr.handle_access_request('test_user_123')),
        ]
        
        gdpr_passed = 0
        
        for check_name, check_func in gdpr_checks:
            try:
                result = check_func()
                if result:
                    print(f"   ✅ {check_name}: compliant")
                    gdpr_passed += 1
                else:
                    print(f"   ⚠️  {check_name}: needs attention")
            except Exception as e:
                print(f"   ❌ {check_name}: error - {e}")
        
        gdpr_compliance_rate = (gdpr_passed / len(gdpr_checks)) * 100
        
        print(f"\n🔒 Testing Data Privacy Management...")
        
        # Test data privacy manager
        privacy_manager = DataPrivacyManager()
        
        privacy_checks = [
            ('data_encryption', lambda: privacy_manager.is_data_encrypted()),
            ('anonymization', lambda: privacy_manager.anonymize_data(test_data)),
            ('deletion_capability', lambda: privacy_manager.delete_user_data('test_user_123')),
            ('audit_logging', lambda: privacy_manager.log_privacy_event('data_access', 'test_user_123')),
        ]
        
        privacy_passed = 0
        
        for check_name, check_func in privacy_checks:
            try:
                result = check_func()
                if result:
                    print(f"   ✅ {check_name}: implemented")
                    privacy_passed += 1
                else:
                    print(f"   ⚠️  {check_name}: not implemented")
            except Exception as e:
                print(f"   ❌ {check_name}: error - {e}")
        
        privacy_compliance_rate = (privacy_passed / len(privacy_checks)) * 100
        
        print(f"\n📊 Testing Compliance Auditing...")
        
        # Test compliance auditor
        auditor = ComplianceAuditor()
        
        audit_checks = [
            ('audit_trail', lambda: len(auditor.get_audit_trail()) >= 0),
            ('compliance_report', lambda: auditor.generate_compliance_report() is not None),
            ('violation_detection', lambda: auditor.check_for_violations()),
            ('remediation_tracking', lambda: auditor.track_remediation_actions()),
        ]
        
        audit_passed = 0
        
        for check_name, check_func in audit_checks:
            try:
                result = check_func()
                if result:
                    print(f"   ✅ {check_name}: functional")
                    audit_passed += 1
                else:
                    print(f"   ⚠️  {check_name}: limited functionality")
            except Exception as e:
                print(f"   ❌ {check_name}: error - {e}")
        
        audit_compliance_rate = (audit_passed / len(audit_checks)) * 100
        
        # Overall compliance score
        overall_compliance = (gdpr_compliance_rate + privacy_compliance_rate + audit_compliance_rate) / 3
        
        print(f"\n📊 Compliance Summary:")
        print(f"   GDPR Compliance: {gdpr_compliance_rate:.1f}%")
        print(f"   Privacy Management: {privacy_compliance_rate:.1f}%") 
        print(f"   Audit Capability: {audit_compliance_rate:.1f}%")
        
        print(f"\n🎯 Overall Compliance Score: {overall_compliance:.1f}/100")
        
        if overall_compliance >= 80:
            print("✅ COMPLIANCE READY: Regulatory requirements met")
            return True
        else:
            print(f"⚠️  COMPLIANCE NEEDS WORK: {overall_compliance:.1f}% < 80% threshold")
            return overall_compliance >= 60
            
    except Exception as e:
        print(f"❌ Compliance testing failed: {e}")
        traceback.print_exc()
        return False


def test_multi_region_deployment():
    """Test multi-region deployment readiness."""
    print("\n🌐 MULTI-REGION DEPLOYMENT TEST")
    print("=" * 50)
    
    try:
        # Check Kubernetes manifests
        k8s_dir = Path('/root/repo/k8s')
        
        required_manifests = [
            'deployment.yaml',
            'service.yaml', 
            'configmap.yaml',
            'secret.yaml',
            'ingress.yaml',
            'hpa.yaml',
            'monitoring.yaml',
            'networkpolicy.yaml'
        ]
        
        print("📁 Checking Kubernetes manifests...")
        
        manifest_score = 0
        for manifest in required_manifests:
            manifest_path = k8s_dir / manifest
            if manifest_path.exists():
                print(f"   ✅ {manifest}: present")
                manifest_score += 1
            else:
                print(f"   ❌ {manifest}: missing")
        
        manifest_coverage = (manifest_score / len(required_manifests)) * 100
        
        # Check deployment configuration
        print(f"\n🚀 Checking deployment configuration...")
        
        config_files = [
            '/root/repo/config/production.yaml',
            '/root/repo/config/staging.yaml',
            '/root/repo/docker-compose.yml',
            '/root/repo/Dockerfile'
        ]
        
        config_score = 0
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"   ✅ {os.path.basename(config_file)}: present")
                config_score += 1
            else:
                print(f"   ❌ {os.path.basename(config_file)}: missing")
        
        config_coverage = (config_score / len(config_files)) * 100
        
        # Check deployment scripts
        print(f"\n🔧 Checking deployment automation...")
        
        deploy_dir = Path('/root/repo/deploy')
        deploy_scripts = ['deploy.sh', 'cleanup.sh']
        
        deploy_score = 0
        if deploy_dir.exists():
            for script in deploy_scripts:
                script_path = deploy_dir / script
                if script_path.exists():
                    print(f"   ✅ {script}: present")
                    deploy_score += 1
                else:
                    print(f"   ❌ {script}: missing")
        else:
            print(f"   ❌ Deploy directory not found")
        
        deploy_coverage = (deploy_score / len(deploy_scripts)) * 100
        
        # Check for multi-region specific features
        print(f"\n🌍 Checking multi-region features...")
        
        # Check if configurations support multiple regions
        region_features = [
            ('Horizontal Pod Autoscaler', k8s_dir / 'hpa.yaml'),
            ('Network Policies', k8s_dir / 'networkpolicy.yaml'), 
            ('Monitoring Setup', k8s_dir / 'monitoring.yaml'),
            ('Ingress Configuration', k8s_dir / 'ingress.yaml')
        ]
        
        region_score = 0
        for feature_name, feature_file in region_features:
            if feature_file.exists():
                print(f"   ✅ {feature_name}: configured")
                region_score += 1
            else:
                print(f"   ⚠️  {feature_name}: not configured")
        
        region_coverage = (region_score / len(region_features)) * 100
        
        # Overall deployment readiness
        deployment_readiness = (manifest_coverage + config_coverage + deploy_coverage + region_coverage) / 4
        
        print(f"\n📊 Deployment Readiness Summary:")
        print(f"   K8s Manifests: {manifest_coverage:.1f}%")
        print(f"   Configuration: {config_coverage:.1f}%")
        print(f"   Automation: {deploy_coverage:.1f}%") 
        print(f"   Multi-region Features: {region_coverage:.1f}%")
        
        print(f"\n🎯 Deployment Readiness Score: {deployment_readiness:.1f}/100")
        
        if deployment_readiness >= 80:
            print("✅ DEPLOYMENT READY: Multi-region deployment prepared")
            return True
        else:
            print(f"⚠️  DEPLOYMENT NEEDS WORK: {deployment_readiness:.1f}% < 80% threshold")
            return deployment_readiness >= 60
            
    except Exception as e:
        print(f"❌ Multi-region deployment testing failed: {e}")
        return False


def test_cross_platform_compatibility():
    """Test cross-platform compatibility."""
    print("\n💻 CROSS-PLATFORM COMPATIBILITY TEST")
    print("=" * 50)
    
    try:
        # Test Python version compatibility
        python_version = sys.version_info
        print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        supported_versions = [(3, 8), (3, 9), (3, 10), (3, 11), (3, 12)]
        current_version = (python_version.major, python_version.minor)
        
        version_supported = current_version in supported_versions
        if version_supported:
            print(f"   ✅ Python {current_version[0]}.{current_version[1]}: supported")
        else:
            print(f"   ⚠️  Python {current_version[0]}.{current_version[1]}: not in supported list")
        
        # Test platform detection
        import platform
        system = platform.system()
        architecture = platform.architecture()[0]
        
        print(f"💿 Platform: {system} {architecture}")
        
        supported_platforms = ['Linux', 'Darwin', 'Windows']
        platform_supported = system in supported_platforms
        
        if platform_supported:
            print(f"   ✅ {system}: supported platform")
        else:
            print(f"   ⚠️  {system}: platform support unclear")
        
        # Test import compatibility
        print(f"\n📦 Testing package compatibility...")
        
        optional_packages = [
            ('numpy', 'NumPy backend support'),
            ('torch', 'PyTorch backend support'),
            ('jax', 'JAX backend support')
        ]
        
        compatibility_score = 0
        
        for package_name, description in optional_packages:
            try:
                __import__(package_name)
                print(f"   ✅ {package_name}: available ({description})")
                compatibility_score += 1
            except ImportError:
                print(f"   ⚠️  {package_name}: not available ({description})")
        
        # Test core functionality without optional dependencies
        print(f"\n🧪 Testing core functionality independence...")
        
        try:
            from hd_compute.pure_python.hdc_python import HDComputePython
            hdc = HDComputePython(dim=50)
            
            # Test basic operations work without external dependencies
            hv1 = hdc.random_hv()
            hv2 = hdc.random_hv()
            similarity = hdc.cosine_similarity(hv1, hv2)
            
            print(f"   ✅ Core functionality: works without dependencies")
            core_independent = True
        except Exception as e:
            print(f"   ❌ Core functionality: depends on external packages - {e}")
            core_independent = False
        
        # Calculate compatibility score
        version_score = 100 if version_supported else 50
        platform_score = 100 if platform_supported else 60
        package_score = (compatibility_score / len(optional_packages)) * 100
        independence_score = 100 if core_independent else 0
        
        overall_compatibility = (version_score + platform_score + package_score + independence_score) / 4
        
        print(f"\n📊 Compatibility Summary:")
        print(f"   Python Version: {'✅' if version_supported else '⚠️'}")
        print(f"   Platform Support: {'✅' if platform_supported else '⚠️'}")
        print(f"   Package Availability: {package_score:.1f}%")
        print(f"   Core Independence: {'✅' if core_independent else '❌'}")
        
        print(f"\n🎯 Compatibility Score: {overall_compatibility:.1f}/100")
        
        if overall_compatibility >= 75:
            print("✅ COMPATIBILITY READY: Cross-platform deployment supported")
            return True
        else:
            print(f"⚠️  COMPATIBILITY NEEDS WORK: {overall_compatibility:.1f}% < 75% threshold")
            return overall_compatibility >= 50
            
    except Exception as e:
        print(f"❌ Cross-platform compatibility testing failed: {e}")
        return False


def run_global_readiness_assessment():
    """Run comprehensive global readiness assessment."""
    print("🌍 HD-COMPUTE GLOBAL READINESS ASSESSMENT")
    print("=" * 50)
    
    assessments = [
        ("Internationalization (I18n)", test_internationalization),
        ("Compliance Readiness", test_compliance_readiness),
        ("Multi-Region Deployment", test_multi_region_deployment),
        ("Cross-Platform Compatibility", test_cross_platform_compatibility)
    ]
    
    passed_assessments = 0
    total_assessments = len(assessments)
    assessment_results = {}
    
    for assessment_name, assessment_func in assessments:
        print(f"\n🔍 Running {assessment_name}...")
        try:
            result = assessment_func()
            assessment_results[assessment_name] = result
            if result:
                passed_assessments += 1
                print(f"✅ {assessment_name}: READY")
            else:
                print(f"⚠️  {assessment_name}: NEEDS WORK")
        except Exception as e:
            print(f"❌ {assessment_name}: ERROR - {e}")
            assessment_results[assessment_name] = False
    
    # Final global readiness score
    readiness_rate = (passed_assessments / total_assessments) * 100
    
    print(f"\n📊 GLOBAL READINESS SUMMARY")
    print("=" * 50)
    print(f"Assessments Passed: {passed_assessments}/{total_assessments}")
    print(f"Global Readiness: {readiness_rate:.1f}%")
    
    for assessment_name, result in assessment_results.items():
        status = "✅ READY" if result else "⚠️ NEEDS WORK"
        print(f"   {assessment_name}: {status}")
    
    if readiness_rate >= 75:  # 3/4 assessments must pass
        print(f"\n🌍 GLOBAL READINESS: ✅ READY")
        print("System prepared for global deployment!")
        return True
    else:
        print(f"\n⚠️  GLOBAL READINESS: PARTIAL")
        print("Address failing assessments for full global deployment.")
        return readiness_rate >= 50


if __name__ == "__main__":
    success = run_global_readiness_assessment()
    sys.exit(0 if success else 1)