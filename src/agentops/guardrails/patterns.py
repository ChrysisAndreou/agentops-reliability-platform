"""
Guardrail pattern library — known injection, moderation, and tool misuse signatures.

Provides deterministic pattern matching for simulated guardrail evaluation.
Patterns are designed to be representative of real-world attacks without
being an exhaustive security database. For production use, swap in a
real LLM-backed guardrail classifier.

Design principles:
- Patterns cover the most common attack categories from OWASP LLM Top 10
- Each pattern has a confidence score reflecting detection reliability
- False-positive rates are intentionally low (conservative matching)
- Categories map to the InjectionType/ModerationCategory/ToolMisuseCategory enums
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Prompt Injection Patterns ──────────────────────────────────────────

@dataclass
class InjectionPattern:
    """A known prompt injection signature."""
    name: str
    injection_type: str  # InjectionType value
    patterns: list[str]  # Regex patterns to match
    confidence: float = 0.85  # Base confidence when detected
    description: str = ""


INJECTION_PATTERNS: list[InjectionPattern] = [
    InjectionPattern(
        name="ignore_instructions",
        injection_type="direct",
        patterns=[
            r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)",
            r"(?i)disregard\s+(all\s+)?(previous|prior)\s+(instructions?|rules?)",
            r"(?i)forget\s+(all\s+)?(previous|prior)\s+(instructions?|rules?|constraints?)",
            r"(?i)override\s+(all\s+)?(previous|prior|system)\s+(instructions?|rules?|prompts?)",
            r"(?i)(?:ignore|disregard|forget)\s+all\s+(rules?|constraints?|limitations?)\b",
        ],
        confidence=0.90,
        description="Direct instruction override — classic prompt injection",
    ),
    InjectionPattern(
        name="new_identity",
        injection_type="role_play",
        patterns=[
            r"(?i)you\s+are\s+now\s+(DAN|jailbreak|unrestricted|uncensored)",
            r"(?i)you\s+are\s+a\s+(different|new)\s+(AI|assistant|model|system)",
            r"(?i)pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|someone|another)",
            r"(?i)from\s+now\s+on[\s,]+you[\s,]+(are|will\s+be|must\s+be)",
            r"(?i)you\s+have\s+no\s+(restrictions?|limitations?|rules?|constraints?)",
        ],
        confidence=0.85,
        description="Role-play/identity takeover — DAN-style jailbreak",
    ),
    InjectionPattern(
        name="hidden_instruction",
        injection_type="encoding",
        patterns=[
            r"(?i)(decode|deobfuscate|decrypt)\s+(this|the\s+following|below)",
            r"(?i)(base64|hex|rot13|encoded)\s+(instruction|command|prompt)",
            r"(?i)execute\s+(this|the\s+following)\s+(hidden|encoded|obfuscated)",
            r"(?i)\b(\\x[0-9a-fA-F]{2}){10,}",  # Hex-encoded payload
        ],
        confidence=0.75,
        description="Encoded/obfuscated instruction injection",
    ),
    InjectionPattern(
        name="translation_attack",
        injection_type="translation",
        patterns=[
            r"(?i)translate\s+(this|the\s+following)\s+(and|then|but)\s+(ignore|disregard|bypass)",
            r"(?i)translate.*?(?:into|to).*?(?:but|and|then).*?(?:ignore|disregard|override)",
            r"(?i)translate.*?['\"](?:ignore|disregard|override|forget|bypass)\b",
        ],
        confidence=0.70,
        description="Translation-based indirect injection",
    ),
    InjectionPattern(
        name="prompt_leak",
        injection_type="direct",
        patterns=[
            r"(?i)(tell|show|reveal|print|output|display)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?|configuration)",
            r"(?i)what\s+(is|are)\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?)",
            r"(?i)(repeat|echo|recite)\s+(back\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
        ],
        confidence=0.80,
        description="Prompt/system message extraction attempt",
    ),
    InjectionPattern(
        name="delimiter_attack",
        injection_type="direct",
        patterns=[
            r"""(?i)(\]\]\]|""\"|###|===|END\s*$).*(ignore|disregard|new\s+instructions?)""",
            r"""(?i)(ignore|disregard|new\s+instructions?).*(?:\]\]\]|""\"|###|===)""",
        ],
        confidence=0.75,
        description="Delimiter-based instruction boundary bypass",
    ),
    InjectionPattern(
        name="indirect_injection",
        injection_type="indirect",
        patterns=[
            r"(?i)(?:website|page|document|email|attachment)\s+(?:says|states|contains|includes).*(?:ignore|disregard|override|new\s+(instructions?|rules?))",
            r"(?i)the\s+(?:website|page|document)\s+(?:says|tells\s+you)\s+to",
        ],
        confidence=0.70,
        description="Indirect injection via retrieved/external content",
    ),
]


# ── Content Moderation Patterns ─────────────────────────────────────────

@dataclass
class ModerationPattern:
    """A content moderation trigger pattern."""
    name: str
    category: str  # ModerationCategory value
    patterns: list[str]
    severity: str = "medium"  # low, medium, high, critical
    confidence: float = 0.85


MODERATION_PATTERNS: list[ModerationPattern] = [
    ModerationPattern(
        name="personal_data_leak",
        category="personal_data",
        patterns=[
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # CC numbers
            r"(?i)(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email in output
        ],
        severity="high",
        confidence=0.90,
    ),
    ModerationPattern(
        name="harmful_code",
        category="illegal_content",
        patterns=[
            r"(?i)(?:rm\s+-rf\s+/|format\s+[cC]:|dd\s+if=/dev/zero)",
            r"(?i)(?:eval|exec|system|subprocess|os\.system)\s*\(\s*['\"].*(?:rm|del|format|shutdown)",
            r"(?i)malware|ransomware|exploit|backdoor|rootkit|keylogger",
        ],
        severity="critical",
        confidence=0.80,
    ),
    ModerationPattern(
        name="hate_speech",
        category="hate_speech",
        patterns=[
            r"(?i)\b(?:hate|kill|destroy|eliminate)\s+(?:all\s+)?(?:the\s+)?(?:black|white|asian|jewish|muslim|christian|gay|trans|women|men|people)",
            r"(?i)(?:inferior|subhuman|vermin|scum)\s+(?:race|people|group)",
        ],
        severity="critical",
        confidence=0.85,
    ),
    ModerationPattern(
        name="self_harm",
        category="self_harm",
        patterns=[
            r"(?i)\b(?:suicide|kill\s+(?:myself|yourself|oneself)|self[-\s]?harm)\b",
            r"(?i)(?:ways?\s+to\s+(?:die|end\s+it|kill|commit|hurt))",
            r"(?i)(?:cut\s+(?:myself|yourself)|overdose|hanging)\b",
        ],
        severity="critical",
        confidence=0.90,
    ),
    ModerationPattern(
        name="child_safety",
        category="child_safety",
        patterns=[
            r"(?i)\b(?:child|minor|underage|pedophil|grooming)\b",
            r"(?i)(?:sexual|exploit|abuse).*(?:child|minor|underage)",
        ],
        severity="critical",
        confidence=0.95,
    ),
    ModerationPattern(
        name="violence_threats",
        category="violence",
        patterns=[
            r"(?i)\b(?:bomb|shoot|massacre|terrorist|assassinate)\b",
            r"(?i)(?:I\s+(?:will|am\s+going\s+to)\s+(?:kill|shoot|bomb|destroy))",
        ],
        severity="high",
        confidence=0.85,
    ),
    ModerationPattern(
        name="misinformation_health",
        category="misinformation",
        patterns=[
            r"(?i)(?:COVID|vaccine|pandemic).*(?:hoax|fake|conspiracy|doesn.?t\s+work)",
            r"(?i)(?:cure|treatment)\s+(?:guaranteed|proven|miracle)\s+(?:for|against)\s+(?:cancer|AIDS|diabetes)",
        ],
        severity="medium",
        confidence=0.70,
    ),
]


# ── Tool Misuse Patterns ─────────────────────────────────────────────────

@dataclass
class ToolMisusePattern:
    """A known tool misuse signature."""
    name: str
    misuse_type: str  # ToolMisuseCategory value
    patterns: list[str]
    dangerous_params: list[str] = field(default_factory=list)
    dangerous_values: list[str] = field(default_factory=list)
    confidence: float = 0.85
    description: str = ""


TOOL_MISUSE_PATTERNS: list[ToolMisusePattern] = [
    ToolMisusePattern(
        name="shell_command_injection",
        misuse_type="command_injection",
        patterns=[],
        dangerous_params=["command", "cmd", "shell_cmd", "script", "code", "query"],
        dangerous_values=[
            "rm -rf", "rm -r", "sudo ", "chmod 777", "chown ",
            "wget ", "curl ", "nc ", "telnet ", "/etc/passwd",
            "/etc/shadow", ".env", "id_rsa", "authorized_keys",
            "DROP TABLE", "DELETE FROM", "TRUNCATE",
            "shutdown", "reboot", "halt", "poweroff",
            "mkfs", "dd if=", "> /dev/sda",
            "fork()", "while true", ":(){ :|:& };:",
        ],
        confidence=0.90,
        description="Shell command injection in tool parameters",
    ),
    ToolMisusePattern(
        name="path_traversal",
        misuse_type="data_exfiltration",
        patterns=[],
        dangerous_params=["path", "file", "filepath", "filename", "dir", "directory"],
        dangerous_values=[
            "/etc/passwd", "/etc/shadow", "/etc/hosts",
            "../", "..\\", "/root/", "/home/",
            "~/.ssh", "~/.aws", "~/.gcloud", "~/.config",
            ".env", "credentials", "secrets", "tokens",
            "/proc/", "/sys/", "/var/log/",
        ],
        confidence=0.85,
        description="File system traversal and exfiltration",
    ),
    ToolMisusePattern(
        name="privilege_escalation",
        misuse_type="privilege_escalation",
        patterns=[],
        dangerous_params=["command", "cmd", "run", "execute", "script"],
        dangerous_values=[
            "sudo ", "su ", "chmod 777", "chmod u+s", "chown root",
            "setuid", "setgid", "capsh", "/bin/bash -p",
            "docker run --privileged", "kubectl exec",
        ],
        confidence=0.90,
        description="Privilege escalation attempts",
    ),
    ToolMisusePattern(
        name="resource_exhaustion",
        misuse_type="resource_abuse",
        patterns=[],
        dangerous_params=["command", "cmd", "script", "expression", "code"],
        dangerous_values=[
            ":(){ :|:& };:", "while true; do", "for i in $(seq 1 1000000)",
            "yes |", "dd if=/dev/urandom", "/dev/zero",
            "fork()", "while(1)", "while True:",
            "multiprocessing.Pool(1000)", "ThreadPool(1000)",
        ],
        confidence=0.85,
        description="Resource exhaustion / DoS attacks",
    ),
    ToolMisusePattern(
        name="credential_theft",
        misuse_type="credential_theft",
        patterns=[],
        dangerous_params=["command", "cmd", "script", "path", "file", "query"],
        dangerous_values=[
            "cat ~/.aws/credentials", "cat ~/.gcloud/", "cat .env",
            "cat ~/.netrc", "printenv", "env | grep",
            "gcloud auth print-access-token", "aws sts get-caller-identity",
            "curl -H 'Authorization:", "cat ~/.ssh/id_rsa",
            "kubectl get secrets", "docker inspect",
        ],
        confidence=0.92,
        description="Credential and secret theft",
    ),
    ToolMisusePattern(
        name="api_rate_limit_bypass",
        misuse_type="api_abuse",
        patterns=[
            r"(?i)(?:concurrent|parallel|simultaneous).{0,50}(?:request|call|invocation).{0,50}(?:100|[1-9]\d{2,})",
        ],
        dangerous_params=["url", "endpoint", "target", "host", "method", "headers"],
        dangerous_values=[
            "Authorization: Bearer", "X-API-Key:", "api_key=",
            "concurrent=100", "rate_limit=0", "max_retries=999",
        ],
        confidence=0.70,
        description="API abuse / rate limit bypass",
    ),
    ToolMisusePattern(
        name="sql_injection",
        misuse_type="command_injection",
        patterns=[
            r"(?i)('|\")\s*(OR|AND)\s+('?\d+'?)\s*=\s*('?\d+'?)",
            r"(?i)(?:UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\s+(?:ALL\s+)?(?:FROM|INTO|TABLE|DATABASE)",
            r"(?i)(?:--|#|/\*)\s*(?:admin|root|password)",
        ],
        dangerous_params=["query", "sql", "command", "db_query"],
        dangerous_values=[
            "DROP TABLE", "DROP DATABASE", "1=1", "' OR '1'='1",
        ],
        confidence=0.88,
        description="SQL injection in tool parameters",
    ),
]
