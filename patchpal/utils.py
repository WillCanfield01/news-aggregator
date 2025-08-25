# patchpal/utils.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple

_WORDS = re.compile(r"[a-z0-9+.#/-]+")

def _tokens(text: str) -> set[str]:
    return set(_WORDS.findall((text or "").lower()))

def _text(item: Dict[str, Any]) -> str:
    return " ".join([
        str(item.get("title") or ""),
        str(item.get("summary") or item.get("content") or ""),
        str(item.get("vendor_guess") or ""),
        " ".join(item.get("tags") or []),
    ])

def _guess_vendor(item: Dict[str, Any]) -> str:
    t = _tokens(_text(item))
    # order matters (more specific first)
    maps = {
        "trend micro apex one": {"apex", "apexone", "apex-one"},
        "n-able n-central": {"n-able", "ncentral", "n-able n-central", "n-able ncentral"},
        "microsoft": {"microsoft", "windows", "edge", "exchange", "office", "msrc"},
        "apple": {"apple", "ios", "ipados", "macos", "watchos"},
        "google chrome": {"chrome", "google chrome"},
        "android": {"android"},
        "adobe": {"adobe", "acrobat", "reader"},
        "zoom": {"zoom"},
        "openssl": {"openssl"},
        "cisco": {"cisco"},
        "fortinet": {"fortinet", "fortigate"},
        "vmware": {"vmware"},
        "linux": {"linux", "ubuntu", "debian", "rhel", "centos", "almalinux", "suse"},
        "nginx": {"nginx"},
        "apache httpd": {"apache http server", "apache httpd", "httpd"},
    }
    for vendor, keys in maps.items():
        if t & keys:
            return vendor
    # fallback to vendor_guess/title first token
    vg = (item.get("vendor_guess") or "").strip()
    if vg:
        return vg.lower()
    title = (item.get("title") or "").lower()
    return title.split("—", 1)[0].strip() or "general"

def _severity_badge(item: Dict[str, Any]) -> str:
    # Prefer explicit severity/CVSS if present; otherwise derive
    sev = str(item.get("severity") or "").upper()
    cvss = item.get("cvss")
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = item.get("epss")
    epss_str = ""
    if isinstance(epss, (int, float)):
        pct = round(float(epss), 2)
        epss_str = f" • EPSS {pct:.2f}"
    if kev:
        kev_str = " • *KEV*"
    else:
        kev_str = ""
    if sev:
        label = sev
    elif isinstance(cvss, (int, float)):
        score = float(cvss)
        if score >= 9.0: label = "CRITICAL"
        elif score >= 7.0: label = "HIGH"
        elif score >= 4.0: label = "MEDIUM"
        else: label = "LOW"
    elif kev or (isinstance(epss, (int, float)) and epss >= 0.5):
        label = "HIGH"
    else:
        label = "MEDIUM"
    emoji = ":rotating_light:" if label in ("CRITICAL", "HIGH") else ":warning:"
    return f"{emoji} *{label}*{epss_str}{kev_str}"

def _docs_links(vendor: str, item: Dict[str, Any]) -> List[Tuple[str,str]]:
    """
    Returns list of (title, url) docs. We include vendor-agnostic links if nothing matches.
    """
    links: List[Tuple[str, str]] = []

    def add(title: str, url: str):
        if not url or not isinstance(url, str): return
        u = url.strip()
        if not (u.startswith("http://") or u.startswith("https://")): return
        for _, existing in links:
            if existing == u:
                return
        links.append((title, u))

    vendor = (vendor or "").lower()

    # Pull any advisory/reference URL shipped with the item
    for key in ("advisory_url", "url", "source_url"):
        if item.get(key):
            add("Advisory", str(item[key]))
            break
    for ref in (item.get("references") or item.get("links") or []):
        if isinstance(ref, dict):
            add(ref.get("title") or "Reference", ref.get("url") or "")
        elif isinstance(ref, str):
            add("Reference", ref)

    # Curated docs by vendor/platform
    if "microsoft" in vendor or "edge" in vendor or "windows" in vendor:
        add("Windows Update (endpoints)", "https://aka.ms/WindowsUpdate")
        add("WSUS: Approve updates", "https://learn.microsoft.com/windows/deployment/update/waas-manage-updates-wsus")
        add("Intune: Force Windows updates", "https://learn.microsoft.com/mem/intune/protect/windows-update-settings")
    elif "apple" in vendor:
        add("How to update iPhone/iPad", "https://support.apple.com/en-us/HT204204")
        add("How to update macOS", "https://support.apple.com/en-us/HT201541")
        add("MDM: Schedule macOS updates", "https://support.apple.com/guide/deployment/de9c2d9a8a3/web")
    elif "google chrome" in vendor or "chrome" in vendor:
        add("Chrome: Force update", "https://support.google.com/chrome/a/answer/6350036")
        add("Chrome release notes", "https://chromereleases.googleblog.com/")
    elif "android" in vendor:
        add("Android: Check for updates", "https://support.google.com/android/answer/7680439")
    elif "adobe" in vendor:
        add("Acrobat/Reader: Update", "https://helpx.adobe.com/acrobat/kb/acrobat-dc-downloads.html")
        add("Adobe Security Bulletins", "https://helpx.adobe.com/security.html")
    elif "trend micro apex one" in vendor or "apex" in vendor:
        add("Apex One: Patch instructions", "https://success.trendmicro.com/solution/1112223")
        add("Apex One: Download Center", "https://docs.trendmicro.com/en-us/business/products/apex-one/apex-one-as-a-service/")
    elif "n-able n-central" in vendor or "ncentral" in vendor:
        add("N-central: Update/Release notes", "https://documentation.n-able.com/remote-management/troubleshooting/Content/Release_Notes/ReleaseNotes.htm")
    elif "openssl" in vendor:
        add("OpenSSL security", "https://www.openssl.org/news/vulnerabilities.html")
    elif "linux" in vendor or "ubuntu" in vendor or "debian" in vendor or "rhel" in vendor or "suse" in vendor:
        add("Linux: General patching", "https://www.cyberciti.biz/faq/how-to-update-all-packages-in-linux/")
    elif "cisco" in vendor:
        add("Cisco PSIRT Advisories", "https://sec.cloudapps.cisco.com/security/center/publicationListing.x")
    elif "fortinet" in vendor:
        add("Fortinet PSIRT", "https://www.fortiguard.com/psirt")
    elif "vmware" in vendor:
        add("VMware Security Advisories", "https://www.vmware.com/security/advisories.html")
    elif "zoom" in vendor:
        add("Zoom: Update clients", "https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0063856")

    # Fallback generic docs if we still have none
    if not links:
        add("How to apply software updates", "https://www.cisa.gov/resources-tools/resources/applying-security-patches")
    return links[:3]

def _action_playbook(vendor: str, item: Dict[str, Any], tone: str) -> List[str]:
    """
    Returns bullet points for "Do this now" (simple or detailed).
    """
    vendor = (vendor or "").lower()
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = float(item.get("epss") or 0.0)
    urgent = kev or epss >= 0.5

    def bullets(lines: List[str]) -> List[str]:
        return [ln for ln in lines if ln.strip()]

    if "microsoft" in vendor or "windows" in vendor or "edge" in vendor:
        if tone == "detailed":
            return bullets([
                "Run *Windows Update* on endpoints and servers; install all available security updates.",
                "If you manage patches: *WSUS* → approve & deploy; or *Intune* → set `Deadline` and force restart.",
                "Reboot systems as required and re-scan for missing KBs.",
                "For internet-facing or high-risk assets, prioritize patching within 24–48 hours.",
            ])
        return bullets([
            "Install latest *Windows*/*Edge* security updates.",
            "If using WSUS/Intune, approve/force deployment.",
            "Reboot if required; re-scan.",
        ])
    if "apple" in vendor:
        if tone == "detailed":
            return bullets([
                "Update *iOS/iPadOS/macOS* to the latest version.",
                "If using MDM, push a *scheduled OS update* to managed devices.",
                "If Rapid Security Response is available, apply it immediately.",
            ])
        return bullets([
            "Update iPhone/iPad/Mac to the latest version.",
            "MDM: push/schedule updates.",
        ])
    if "google chrome" in vendor or "chrome" in vendor:
        if tone == "detailed":
            return bullets([
                "Update Chrome to the latest stable version.",
                "Admin: enforce *AutoUpdate* and force relaunch; disable outdated versions.",
            ])
        return bullets([
            "Update Chrome and relaunch the browser.",
        ])
    if "adobe" in vendor:
        return bullets([
            "Update *Adobe Acrobat/Reader* to the latest build.",
            "If enterprise managed, push updates via your software management tool.",
        ])
    if "trend micro apex one" in vendor or "apex" in vendor:
        return bullets([
            "Apply the latest *Apex One* patch/hotfix on the server/console.",
            "Update agents/endpoints from the console and verify version.",
        ])
    if "n-able n-central" in vendor or "ncentral" in vendor:
        return bullets([
            "Upgrade *N-central* to the latest release.",
            "Follow vendor hardening guidance and verify exposed interfaces are restricted.",
        ])
    if "openssl" in vendor or "linux" in vendor:
        return bullets([
            "Update OpenSSL/OS packages via your distro package manager (`apt`, `yum`, `dnf`, `zypper`).",
            "Restart dependent services if required and re-scan.",
        ])
    if "cisco" in vendor or "fortinet" in vendor or "vmware" in vendor:
        return bullets([
            "Apply the vendor’s fixed version/patch on affected appliances.",
            "Restrict management interfaces; back up config before upgrade.",
        ])
    if "zoom" in vendor:
        return bullets([
            "Update Zoom desktop and mobile clients to the latest version.",
            "Admin: enforce auto-update policy in the Zoom admin portal.",
        ])

    # Generic fallback
    if tone == "detailed":
        return bullets([
            "Apply the vendor’s latest *security update*.",
            "Prioritize internet-exposed and high-value systems.",
            "Reboot or restart services if required; verify remediation with a scan.",
        ])
    return bullets([
        "Install the vendor’s latest security update.",
        "Prioritize internet-exposed assets.",
    ])

def _build_title(item: Dict[str, Any], idx: int) -> str:
    title = str(item.get("title") or "")
    if title:
        # Ensure index prefix exists
        if not title.startswith(f"{idx})"):
            return f"*{idx}) {title}*"
        return f"*{title}*"
    # fallback
    cve = ""
    m = re.search(r"(CVE-\d{4}-\d+)", title or "")
    if not m:
        m = re.search(r"(CVE-\d{4}-\d+)", str(item.get("summary") or ""))
    if m:
        cve = m.group(1)
    return f"*{idx}) {cve or 'Security update'}*"

def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    """
    Returns Slack mrkdwn for one item:
      Title
      <severity badges>
      What happened: <plain sentence>
      Do this now: • bullets
      Docs: <link> · <link> · <link>
    """
    tone = (tone or "simple").lower()
    vendor = _guess_vendor(item)
    severity = _severity_badge(item)

    summary = str(item.get("summary") or item.get("content") or "").strip()
    # Make the first sentence cleaner for non-engineers
    summary = re.sub(r"\s+", " ", summary)
    if len(summary) > 600:
        summary = summary[:600].rstrip() + "…"

    actions = _action_playbook(vendor, item, tone)
    docs = _docs_links(vendor, item)

    lines: List[str] = []
    lines.append(_build_title(item, idx))
    lines.append(f"{severity}")
    if summary:
        lines.append(f"*What happened:* {summary}")

    if actions:
        if tone == "detailed":
            lines.append("*Do this now (step-by-step):*")
        else:
            lines.append("*Do this now:*")
        for a in actions:
            lines.append(f"• {a}")

    if docs:
        doc_str = "  ·  ".join([f"<{u}|{t}>" for t, u in docs])
        lines.append(f"*Docs:* {doc_str}")

    text = "\n".join(lines).strip()
    # Slack section hard limit safety; caller also trims
    if len(text) > 2900:
        text = text[:2900] + "…"
    return text
