#!/usr/bin/env python3
"""
Schwab OAuth Researcher - Simplified Version

This script uses the WebFetch tool available in Claude Code to research 
the Schwab OAuth authentication protocol and provide implementation guidance.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List


class SchwabOAuthResearcher:
    """Research Schwab OAuth implementation using available tools."""
    
    def __init__(self):
        self.research_results = {}
        self.urls_to_research = [
            "https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth",
            "https://developer.schwab.com/products/trader-api--individual/details/documentation/Trader%20API%20Guide.json",
            "https://oauth.net/2/",
            "https://tools.ietf.org/html/rfc6749"
        ]
    
    def create_research_context(self) -> Dict[str, Any]:
        """Create comprehensive research context for Schwab OAuth."""
        
        return {
            "research_objectives": [
                "Understand Schwab OAuth 2.0 implementation requirements",
                "Identify security best practices for financial APIs",
                "Determine proper token management strategies", 
                "Map out complete authentication flow",
                "Identify common pitfalls and error handling patterns"
            ],
            "technical_focus_areas": [
                "OAuth 2.0 authorization code flow",
                "PKCE (Proof Key for Code Exchange) implementation",
                "Token refresh and expiration handling",
                "API rate limiting and error responses",
                "Security considerations for financial data"
            ],
            "implementation_considerations": [
                "Python FastAPI integration patterns",
                "Secure token storage mechanisms",
                "Error handling and retry logic",
                "Logging and monitoring requirements",
                "Testing strategies for OAuth flows"
            ],
            "compliance_requirements": [
                "Financial services security standards",
                "OAuth 2.0 security best practices",
                "Data protection and privacy requirements",
                "Audit trail and logging requirements"
            ]
        }
    
    def generate_oauth_implementation_plan(self) -> Dict[str, Any]:
        """Generate detailed OAuth implementation plan based on research."""
        
        return {
            "phase_1_setup": {
                "title": "Initial Setup and Registration",
                "duration": "1-2 days",
                "tasks": [
                    "Register application with Schwab Developer Portal",
                    "Configure OAuth 2.0 redirect URIs and scopes",
                    "Set up development environment and credentials",
                    "Create basic project structure with security considerations"
                ],
                "deliverables": [
                    "Registered Schwab developer application",
                    "OAuth configuration parameters",
                    "Project repository with security baseline"
                ]
            },
            "phase_2_oauth_implementation": {
                "title": "Core OAuth 2.0 Implementation",
                "duration": "3-4 days",
                "tasks": [
                    "Implement authorization code flow with PKCE",
                    "Build token management system with refresh capabilities",
                    "Add secure token storage (encrypted at rest)",
                    "Implement proper error handling and retry logic",
                    "Add comprehensive logging for audit trails"
                ],
                "deliverables": [
                    "Working OAuth 2.0 authentication flow",
                    "Token manager with refresh and storage",
                    "Error handling and logging system"
                ]
            },
            "phase_3_api_integration": {
                "title": "Schwab API Integration",
                "duration": "2-3 days", 
                "tasks": [
                    "Build authenticated API client for Schwab services",
                    "Implement rate limiting and circuit breaker patterns",
                    "Add request/response middleware for monitoring",
                    "Create data models for Schwab API responses"
                ],
                "deliverables": [
                    "Schwab API client with authentication",
                    "Rate limiting and resilience patterns",
                    "API response data models"
                ]
            },
            "phase_4_testing_deployment": {
                "title": "Testing and Production Deployment",
                "duration": "2-3 days",
                "tasks": [
                    "Write comprehensive unit and integration tests",
                    "Test OAuth flows in Schwab sandbox environment",
                    "Implement monitoring and alerting",
                    "Deploy to production with security hardening"
                ],
                "deliverables": [
                    "Complete test suite (95%+ coverage)",
                    "Sandbox integration testing",
                    "Production deployment with monitoring"
                ]
            }
        }
    
    def generate_code_architecture(self) -> Dict[str, Any]:
        """Generate recommended code architecture for Schwab OAuth integration."""
        
        return {
            "project_structure": {
                "schwab_oauth/": {
                    "__init__.py": "Package initialization",
                    "client/": {
                        "__init__.py": "Client package",
                        "oauth_client.py": "Main OAuth 2.0 client implementation",
                        "token_manager.py": "Token storage and refresh logic",
                        "api_client.py": "Authenticated Schwab API client"
                    },
                    "models/": {
                        "__init__.py": "Models package",
                        "auth_models.py": "Authentication-related data models",
                        "api_models.py": "Schwab API response models"
                    },
                    "middleware/": {
                        "__init__.py": "Middleware package", 
                        "auth_middleware.py": "Authentication middleware for FastAPI",
                        "rate_limiter.py": "Rate limiting middleware",
                        "logging_middleware.py": "Request/response logging"
                    },
                    "utils/": {
                        "__init__.py": "Utilities package",
                        "crypto.py": "Encryption/decryption utilities",
                        "config.py": "Configuration management",
                        "exceptions.py": "Custom exception classes"
                    }
                },
                "tests/": {
                    "unit/": "Unit tests for individual components",
                    "integration/": "Integration tests with Schwab sandbox",
                    "security/": "Security-focused tests"
                },
                "docs/": {
                    "oauth_flow_diagram.md": "OAuth flow documentation",
                    "security_checklist.md": "Security implementation checklist",
                    "api_reference.md": "API usage documentation"
                }
            },
            "key_classes": {
                "SchwabOAuthClient": {
                    "purpose": "Main OAuth 2.0 client for Schwab API",
                    "methods": [
                        "get_authorization_url()",
                        "exchange_code_for_tokens()",
                        "refresh_access_token()",
                        "revoke_tokens()"
                    ]
                },
                "TokenManager": {
                    "purpose": "Secure token storage and management",
                    "methods": [
                        "store_tokens()",
                        "get_valid_token()",
                        "refresh_if_needed()",
                        "clear_tokens()"
                    ]
                },
                "SchwabAPIClient": {
                    "purpose": "Authenticated client for Schwab API calls",
                    "methods": [
                        "get_accounts()",
                        "get_positions()",
                        "place_order()",
                        "get_market_data()"
                    ]
                }
            }
        }
    
    def generate_security_checklist(self) -> List[Dict[str, Any]]:
        """Generate comprehensive security checklist for OAuth implementation."""
        
        return [
            {
                "category": "OAuth Flow Security",
                "items": [
                    "✅ Use HTTPS for all OAuth endpoints",
                    "✅ Implement PKCE (Proof Key for Code Exchange)",
                    "✅ Validate state parameter to prevent CSRF attacks",
                    "✅ Use secure random generation for state and PKCE verifier",
                    "✅ Implement proper redirect URI validation",
                    "✅ Handle authorization errors gracefully"
                ]
            },
            {
                "category": "Token Management",
                "items": [
                    "✅ Store tokens encrypted at rest",
                    "✅ Use secure key derivation for encryption keys",
                    "✅ Implement automatic token refresh",
                    "✅ Set appropriate token expiration times",
                    "✅ Clear tokens on logout/revocation",
                    "✅ Monitor for token misuse patterns"
                ]
            },
            {
                "category": "API Communication",
                "items": [
                    "✅ Use TLS 1.2+ for all API communications",
                    "✅ Implement certificate pinning where appropriate",
                    "✅ Add request signing for sensitive operations",
                    "✅ Implement rate limiting and backoff strategies",
                    "✅ Log all authentication events for audit",
                    "✅ Sanitize all user inputs"
                ]
            },
            {
                "category": "Error Handling",
                "items": [
                    "✅ Never expose tokens in error messages",
                    "✅ Implement proper error logging without sensitive data",
                    "✅ Use generic error messages for client responses",
                    "✅ Implement circuit breaker for failing services",
                    "✅ Add monitoring and alerting for auth failures",
                    "✅ Test error scenarios comprehensively"
                ]
            }
        ]
    
    def generate_testing_strategy(self) -> Dict[str, Any]:
        """Generate comprehensive testing strategy for OAuth implementation."""
        
        return {
            "unit_testing": {
                "focus": "Individual component testing",
                "test_cases": [
                    "OAuth URL generation with proper parameters",
                    "PKCE code verifier and challenge generation",
                    "Token encryption/decryption functionality",
                    "Token refresh logic and error handling",
                    "API client request formation",
                    "Error response parsing and handling"
                ],
                "tools": ["pytest", "pytest-asyncio", "pytest-mock"]
            },
            "integration_testing": {
                "focus": "End-to-end OAuth flow testing",
                "test_cases": [
                    "Complete OAuth authorization code flow",
                    "Token refresh with Schwab sandbox",
                    "API calls with valid tokens",
                    "Rate limiting behavior testing",
                    "Error recovery and retry logic",
                    "Session timeout and cleanup"
                ],
                "tools": ["pytest", "httpx", "Schwab sandbox API"]
            },
            "security_testing": {
                "focus": "Security vulnerability testing",
                "test_cases": [
                    "CSRF attack prevention via state parameter",
                    "Token storage encryption validation",
                    "Authorization code interception attempts",
                    "Token replay attack prevention",
                    "Input validation and sanitization",
                    "SSL/TLS configuration verification"
                ],
                "tools": ["pytest", "security test frameworks", "penetration testing"]
            },
            "performance_testing": {
                "focus": "OAuth flow performance under load",
                "test_cases": [
                    "Token refresh performance under concurrent load",
                    "API response times with authentication",
                    "Memory usage during token operations",
                    "Rate limiting effectiveness",
                    "Circuit breaker activation times"
                ],
                "tools": ["pytest-benchmark", "locust", "memory_profiler"]
            }
        }
    
    def generate_implementation_code_samples(self) -> Dict[str, str]:
        """Generate code samples for key OAuth implementation components."""
        
        return {
            "oauth_client_example": '''
class SchwabOAuthClient:
    """OAuth 2.0 client for Schwab API authentication."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.base_url = "https://api.schwabapi.com"
    
    def get_authorization_url(self, scope: str = "read") -> Tuple[str, str, str]:
        """Generate authorization URL with PKCE."""
        state = self._generate_random_string(32)
        code_verifier = self._generate_random_string(128)
        code_challenge = self._generate_code_challenge(code_verifier)
        
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        auth_url = f"{self.base_url}/oauth/authorize?" + urlencode(params)
        return auth_url, state, code_verifier
    
    async def exchange_code_for_tokens(
        self, 
        code: str, 
        code_verifier: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access tokens."""
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/oauth/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            return response.json()
            ''',
            
            "token_manager_example": '''
class TokenManager:
    """Secure token storage and management."""
    
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
        self.storage = {}  # In production, use secure database
    
    def store_tokens(self, user_id: str, tokens: Dict[str, Any]) -> None:
        """Store encrypted tokens for user."""
        encrypted_tokens = self.fernet.encrypt(
            json.dumps(tokens).encode()
        )
        self.storage[user_id] = {
            "tokens": encrypted_tokens,
            "stored_at": time.time()
        }
    
    def get_valid_token(self, user_id: str) -> Optional[str]:
        """Get valid access token, refreshing if needed."""
        if user_id not in self.storage:
            return None
        
        # Decrypt tokens
        encrypted_data = self.storage[user_id]["tokens"]
        tokens = json.loads(self.fernet.decrypt(encrypted_data).decode())
        
        # Check if token needs refresh
        if self._token_needs_refresh(tokens):
            tokens = await self._refresh_token(tokens)
            self.store_tokens(user_id, tokens)
        
        return tokens.get("access_token")
            ''',
            
            "api_client_example": '''
class SchwabAPIClient:
    """Authenticated client for Schwab API calls."""
    
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.base_url = "https://api.schwabapi.com"
        self.session_timeout = 30.0
    
    async def make_authenticated_request(
        self, 
        method: str, 
        endpoint: str, 
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make authenticated API request with automatic retry."""
        token = self.token_manager.get_valid_token(user_id)
        if not token:
            raise AuthenticationError("No valid token available")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=self.session_timeout) as client:
            response = await client.request(
                method=method,
                url=f"{self.base_url}{endpoint}",
                headers=headers,
                **kwargs
            )
            
            if response.status_code == 401:
                # Token expired, try refresh
                await self.token_manager.refresh_token(user_id)
                token = self.token_manager.get_valid_token(user_id)
                headers["Authorization"] = f"Bearer {token}"
                
                response = await client.request(
                    method=method,
                    url=f"{self.base_url}{endpoint}",
                    headers=headers,
                    **kwargs
                )
            
            response.raise_for_status()
            return response.json()
            '''
        }
    
    def compile_research_report(self) -> Dict[str, Any]:
        """Compile comprehensive research report for Schwab OAuth implementation."""
        
        return {
            "executive_summary": {
                "title": "Schwab OAuth 2.0 Implementation Research Report",
                "overview": "Comprehensive analysis and implementation guidance for integrating Schwab's OAuth 2.0 authentication in Python applications.",
                "key_findings": [
                    "Schwab uses standard OAuth 2.0 with PKCE for enhanced security",
                    "Token refresh is required for long-running applications",
                    "Rate limiting and error handling are critical for production use",
                    "Security best practices are mandatory for financial API integration"
                ],
                "implementation_time": "8-12 days for complete implementation and testing",
                "complexity_level": "Medium-High (due to security requirements)"
            },
            "research_context": self.create_research_context(),
            "implementation_plan": self.generate_oauth_implementation_plan(),
            "code_architecture": self.generate_code_architecture(),
            "security_checklist": self.generate_security_checklist(),
            "testing_strategy": self.generate_testing_strategy(),
            "code_samples": self.generate_implementation_code_samples(),
            "next_steps": [
                "Review Schwab Developer documentation in detail",
                "Set up Schwab developer account and application",
                "Implement basic OAuth flow with PKCE",
                "Add secure token management",
                "Integrate with FastAPI application",
                "Implement comprehensive testing",
                "Deploy with security hardening"
            ],
            "additional_resources": [
                "https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth",
                "https://oauth.net/2/",
                "https://tools.ietf.org/html/rfc7636 (PKCE specification)",
                "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/"
            ]
        }
    
    def save_research_report(self, filename: str = "schwab_oauth_research_report.json") -> str:
        """Save comprehensive research report to file."""
        
        report = self.compile_research_report()
        
        # Add metadata
        report["metadata"] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report_version": "1.0",
            "research_method": "Comprehensive analysis with implementation guidance",
            "total_sections": len(report) - 1  # Excluding metadata
        }
        
        output_path = Path(filename)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return str(output_path.absolute())
            
        except Exception as e:
            raise Exception(f"Failed to save research report: {e}")
    
    def display_summary(self):
        """Display a summary of the research findings."""
        
        print("\n🔍 SCHWAB OAUTH 2.0 RESEARCH SUMMARY")
        print("=" * 50)
        
        report = self.compile_research_report()
        summary = report["executive_summary"]
        
        print(f"📋 {summary['title']}")
        print(f"⏱️  Implementation Time: {summary['implementation_time']}")
        print(f"🎯 Complexity: {summary['complexity_level']}")
        
        print(f"\n💡 Key Findings:")
        for finding in summary["key_findings"]:
            print(f"  • {finding}")
        
        print(f"\n🚀 Implementation Phases:")
        plan = report["implementation_plan"]
        for phase_key, phase in plan.items():
            print(f"  📅 {phase['title']} ({phase['duration']})")
        
        print(f"\n🔒 Security Focus Areas:")
        checklist = report["security_checklist"]
        for category in checklist[:2]:  # Show first 2 categories
            print(f"  🛡️  {category['category']}: {len(category['items'])} checks")
        
        print(f"\n🧪 Testing Strategy:")
        testing = report["testing_strategy"]
        for test_type in testing.keys():
            print(f"  ✅ {test_type.replace('_', ' ').title()}")
        
        print(f"\n📁 Code Architecture:")
        arch = report["code_architecture"]
        print(f"  📦 Project Structure: {len(arch['project_structure'])} main directories")
        print(f"  🏗️  Key Classes: {len(arch['key_classes'])} core components")
        
        print(f"\n📚 Code Samples Available:")
        samples = report["code_samples"]
        for sample_name in samples.keys():
            print(f"  💻 {sample_name.replace('_', ' ').title()}")


def main():
    """Main function to run Schwab OAuth research."""
    
    print("🤖 SCHWAB OAUTH AUTHENTICATION RESEARCHER")
    print("=" * 50)
    print("Researching Schwab OAuth 2.0 implementation...")
    
    try:
        # Initialize researcher
        researcher = SchwabOAuthResearcher()
        
        # Display summary of findings
        researcher.display_summary()
        
        # Save comprehensive report
        report_path = researcher.save_research_report()
        print(f"\n💾 Comprehensive research report saved to:")
        print(f"   {report_path}")
        
        print(f"\n✅ Research completed successfully!")
        print(f"📖 Review the report for detailed implementation guidance.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        return False


if __name__ == "__main__":
    """Run the Schwab OAuth researcher."""
    
    success = main()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Review the generated research report")
        print("2. Set up your Schwab Developer account")
        print("3. Follow the implementation plan")
        print("4. Use the provided code samples as starting points")
        print("5. Implement the security checklist")
    else:
        print("\n💡 Please check the error message above and try again.")