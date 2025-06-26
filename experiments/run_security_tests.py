#!/usr/bin/env python3
"""
Security test runner for TradeKnowledge.

This script runs comprehensive security tests and generates a security report.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, capture_output=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def run_security_tests():
    """Run security-specific pytest tests"""
    print("🔒 Running Security Tests...")
    success, stdout, stderr = run_command(
        "python -m pytest tests/security/ -v --tb=short"
    )
    
    if success:
        print("✅ Security tests passed")
    else:
        print("❌ Security tests failed")
        print(stderr)
    
    return success


def run_bandit_scan():
    """Run Bandit security linter"""
    print("🔍 Running Bandit Security Scan...")
    success, stdout, stderr = run_command(
        "bandit -r src/ -f json -o bandit-report.json"
    )
    
    if success:
        print("✅ Bandit scan completed")
        # Parse results
        try:
            with open("bandit-report.json", "r") as f:
                report = json.load(f)
                issues = len(report.get("results", []))
                print(f"   Found {issues} security issues")
        except:
            print("   Could not parse Bandit report")
    else:
        print("❌ Bandit scan failed")
        print(stderr)
    
    return success


def run_safety_check():
    """Run Safety dependency vulnerability check"""
    print("🛡️  Running Safety Dependency Check...")
    success, stdout, stderr = run_command("safety check --json")
    
    if success:
        print("✅ No known vulnerabilities in dependencies")
    else:
        print("❌ Vulnerabilities found in dependencies")
        try:
            vulnerabilities = json.loads(stderr)
            print(f"   Found {len(vulnerabilities)} vulnerabilities")
            for vuln in vulnerabilities[:3]:  # Show first 3
                print(f"   - {vuln.get('advisory', 'Unknown')}")
        except:
            print(f"   {stderr}")
    
    return success


def run_input_validation_tests():
    """Run specific input validation tests"""
    print("🚫 Running Input Validation Tests...")
    success, stdout, stderr = run_command(
        "python -m pytest tests/security/test_injection_attacks.py -v"
    )
    
    if success:
        print("✅ Input validation tests passed")
    else:
        print("❌ Input validation tests failed")
    
    return success


def run_auth_security_tests():
    """Run authentication security tests"""
    print("🔐 Running Authentication Security Tests...")
    success, stdout, stderr = run_command(
        "python -m pytest tests/security/test_auth_security.py -v"
    )
    
    if success:
        print("✅ Authentication security tests passed")
    else:
        print("❌ Authentication security tests failed")
    
    return success


def generate_security_report():
    """Generate comprehensive security report"""
    print("📊 Generating Security Report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": {},
        "recommendations": []
    }
    
    # Run all security checks
    report["test_results"]["security_tests"] = run_security_tests()
    report["test_results"]["bandit_scan"] = run_bandit_scan()
    report["test_results"]["safety_check"] = run_safety_check()
    report["test_results"]["input_validation"] = run_input_validation_tests()
    report["test_results"]["auth_security"] = run_auth_security_tests()
    
    # Calculate overall score
    passed_tests = sum(report["test_results"].values())
    total_tests = len(report["test_results"])
    score = (passed_tests / total_tests) * 100
    
    report["security_score"] = score
    
    # Add recommendations based on results
    if not report["test_results"]["bandit_scan"]:
        report["recommendations"].append("Fix Bandit security issues in source code")
    
    if not report["test_results"]["safety_check"]:
        report["recommendations"].append("Update dependencies with known vulnerabilities")
    
    if not report["test_results"]["input_validation"]:
        report["recommendations"].append("Improve input validation and sanitization")
    
    if not report["test_results"]["auth_security"]:
        report["recommendations"].append("Strengthen authentication and authorization")
    
    # Save report
    with open("security-report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Security Score: {score:.1f}%")
    print(f"📄 Report saved to: security-report.json")
    
    return report


def main():
    """Main security test runner"""
    print("🚀 TradeKnowledge Security Test Suite")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    try:
        report = generate_security_report()
        
        print("\n📈 Security Test Summary:")
        print("-" * 30)
        for test_name, passed in report["test_results"].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:<20} {status}")
        
        print(f"\n🎯 Overall Security Score: {report['security_score']:.1f}%")
        
        if report["recommendations"]:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")
        
        # Exit with error if security score is too low
        if report["security_score"] < 80:
            print(f"\n⚠️  Security score below threshold (80%)")
            sys.exit(1)
        else:
            print(f"\n🎉 Security tests passed!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n💥 Error running security tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()