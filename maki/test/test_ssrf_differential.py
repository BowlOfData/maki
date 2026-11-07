"""Differential and fuzz tests of the SSRF guard, against an oracle that is not the code under test.

The oracle is Python's own ``ipaddress`` classification of what the operating system's resolver returns for a
host string: if the OS would connect to an address that is not publicly routable, maki must refuse the host
(either in the static URL check or when the connect-time resolution is validated). Inputs are

* every IANA special-purpose range (IPv4 and IPv6), first, last and a middle address of each;
* IPv6 forms that embed an IPv4 address (mapped, 6to4, NAT64, IPv4-compatible) and Teredo;
* thousands of spellings of the addresses attackers use (127.0.0.1, 169.254.169.254, 10.0.0.1): decimal, hex,
  octal, mixed-base dotted, short forms, leading zeros, trailing dots, mixed case, userinfo, ports;
* hostnames that resolve to a refused address (rebinding, round-robin with one bad record).

Public addresses must still pass, so the guard is not a blanket refusal.
"""
import ipaddress
import random
import socket
import unittest
from unittest.mock import patch

from maki.connector import _resolve_and_validate, validate_url
from maki.exceptions import MakiError

# (192.88.99.0/24, the deprecated 6to4 relay anycast, is deliberately absent: it is a globally routable address)
V4_SPECIAL = ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
              "192.0.0.0/24", "192.0.2.0/24", "192.168.0.0/16", "198.18.0.0/15",
              "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32"]
# (::ffff:0:0/96 and 64:ff9b::/96 are judged on the IPv4 address they carry, so they are covered by the
# embedded-address tests below rather than refused as ranges: ::ffff:8.8.8.8 reaches a public host)
V6_SPECIAL = ["::/128", "::1/128", "64:ff9b:1::/48", "100::/64", "2001::/23",
              "2001:db8::/32", "2002::/16", "fc00::/7", "fe80::/10", "ff00::/8"]
PUBLIC = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "151.101.1.69", "2606:4700:4700::1111",
          "2001:4860:4860::8888", "2a00:1450:4009:81f::200e"]


def blocked(host: str) -> bool:
    """True when maki would not connect to *host*: the static check or the connect-time check refuses it."""
    try:
        validate_url(f"http://{host}/")
        _resolve_and_validate(host, 80)
    except MakiError:
        return True
    return False


def os_would_reach(host: str):
    """The non-public address the OS resolver maps *host* to, or None (public, or does not resolve)."""
    try:
        infos = socket.getaddrinfo(host.strip("[]"), 80, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        return None
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            return address
    return None


def samples(network: str):
    net = ipaddress.ip_network(network)
    picks = {net[0], net[-1], net[min(5, net.num_addresses - 1)], net[net.num_addresses // 2]}
    return sorted(picks, key=int)


class TestSpecialPurposeRanges(unittest.TestCase):
    def test_every_ipv4_special_range_is_refused(self):
        for network in V4_SPECIAL:
            for ip in samples(network):
                with self.subTest(range=network, ip=str(ip)):
                    self.assertTrue(blocked(str(ip)), f"{ip} ({network}) was let through")

    def test_every_ipv6_special_range_is_refused(self):
        for network in V6_SPECIAL:
            for ip in samples(network):
                with self.subTest(range=network, ip=str(ip)):
                    self.assertTrue(blocked(f"[{ip}]"), f"{ip} ({network}) was let through")

    def test_ipv6_forms_that_embed_a_refused_ipv4_address_are_refused(self):
        embedded = {
            "6to4 of 10.0.0.1": "2002:0a00:0001::1", "6to4 of 127.0.0.1": "2002:7f00:0001::1",
            "6to4 of 169.254.169.254": "2002:a9fe:a9fe::1", "6to4 of 192.168.1.1": "2002:c0a8:0101::",
            "NAT64 of 127.0.0.1": "64:ff9b::7f00:1", "NAT64 of 169.254.169.254": "64:ff9b::a9fe:a9fe",
            "mapped 10.0.0.1": "::ffff:10.0.0.1", "mapped metadata": "::ffff:169.254.169.254",
            "mapped loopback": "::ffff:127.0.0.1", "compat loopback": "::127.0.0.1", "compat 10.0.0.1": "::10.0.0.1",
            "teredo": "2001:0:4136:e378:8000:63bf:3fff:fdd2",
        }
        for name, address in embedded.items():
            with self.subTest(form=name):
                self.assertTrue(blocked(f"[{address}]"), f"{name} [{address}] was let through")

    def test_the_public_internet_is_still_reachable(self):
        for address in PUBLIC:
            host = f"[{address}]" if ":" in address else address
            with self.subTest(address=address):
                self.assertFalse(blocked(host), f"{address} is public but was refused")

    def test_public_ipv4_embedded_in_ipv6_is_judged_on_the_ipv4(self):
        # mapped and NAT64 forms of a public address stay allowed (the embedded address is what is reached)
        for host in ("[::ffff:8.8.8.8]", "[64:ff9b::808:808]"):
            with self.subTest(host=host):
                self.assertFalse(blocked(host), host)

    def test_the_opt_in_for_loopback_opens_loopback_forms_only(self):
        with patch.dict("os.environ", {"MAKI_ALLOW_LOOPBACK_URLS": "1"}):
            for host in ("127.0.0.1", "[::1]", "[::ffff:127.0.0.1]", "[64:ff9b::7f00:1]"):
                with self.subTest(host=host):
                    self.assertFalse(blocked(host), host)
            # 6to4 is refused as a range, so even a 6to4 wrapper of loopback stays shut; nothing private opens
            for host in ("10.0.0.1", "169.254.169.254", "[2002:7f00:1::1]", "[2002:0a00:1::1]", "[64:ff9b::a9fe:a9fe]",
                         "[::ffff:10.0.0.1]"):
                with self.subTest(host=host):
                    self.assertTrue(blocked(host), host)


def spellings(ip: ipaddress.IPv4Address, rng: random.Random):
    """Many ways to write the same IPv4 address that inet_aton-style resolvers accept."""
    n, octets = int(ip), [int(o) for o in str(ip).split(".")]
    forms = {str(n), hex(n), hex(n).upper(), "0" + oct(n)[2:], f"0{n:o}", str(ip), str(ip) + ".", str(ip).upper(),
             f"{octets[0]}.{n & 0xFFFFFF}", f"{octets[0]}.{octets[1]}.{n & 0xFFFF}",
             ".".join(f"0{o:o}" for o in octets), ".".join(hex(o) for o in octets),
             ".".join(f"{o:03d}" for o in octets), f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}.",
             f"[::ffff:{ip}]", f"[::ffff:{n >> 16:x}:{n & 0xFFFF:x}]", f"[::{ip}]", f"[0:0:0:0:0:ffff:{ip}]"}
    for _ in range(12):  # mixed bases octet by octet
        forms.add(".".join(rng.choice([str(o), hex(o), f"0{o:o}", f"{o:03d}"]) for o in octets))
    for _ in range(4):
        forms.add(f"{octets[0]}.{octets[1]}.{(octets[2] << 8) | octets[3]}")
    return sorted(forms)


class TestSpellingsOfDangerousAddresses(unittest.TestCase):
    TARGETS = ["127.0.0.1", "127.255.255.254", "169.254.169.254", "10.0.0.1", "172.16.0.1", "192.168.1.1",
               "100.64.0.1", "0.0.0.1", "224.0.0.1"]

    def test_whatever_the_os_would_connect_to_is_refused_by_maki(self):
        rng = random.Random(2026)
        checked = reached = 0
        for target in self.TARGETS:
            for form in spellings(ipaddress.IPv4Address(target), rng):
                checked += 1
                address = os_would_reach(form)
                if address is None:
                    continue  # the OS would not connect to a non-public address with this spelling
                reached += 1
                with self.subTest(target=target, spelling=form, os_resolves_to=str(address)):
                    self.assertTrue(blocked(form), f"{form!r} reaches {address} but was let through")
        self.assertGreater(checked, 200)
        self.assertGreater(reached, 100, "the oracle found too few reachable spellings to be meaningful")

    def test_random_wrappers_do_not_hide_a_target(self):
        """userinfo, ports, paths, case and trailing dots around a dangerous host never open it."""
        rng = random.Random(7)
        for _ in range(400):
            host = rng.choice(["127.0.0.1", "[::1]", "169.254.169.254", "10.0.0.1", "[::ffff:10.0.0.1]",
                               "[2002:7f00:1::1]", "2130706433", "0x7f000001", "017700000001"])
            userinfo = rng.choice(["", "user@", "user:pw@", "a:b@", "@"])
            port = rng.choice(["", ":80", ":8080", ":443", ":11434"])
            url = f"{rng.choice(['http', 'https', 'HTTP', 'Http'])}://{userinfo}{host}{port}/{rng.choice(['', 'x', 'a/b?q=1#f'])}"
            with self.subTest(url=url):
                try:
                    validate_url(url)
                    parsed_host = url.split("@")[-1].split("/")[0]
                    parsed_host = parsed_host.rsplit(":", 1)[0] if parsed_host.count(":") == 1 or parsed_host.endswith("]") is False and ":" in parsed_host[parsed_host.find("]") + 1:] else parsed_host
                    _resolve_and_validate(parsed_host.strip(), int(port[1:]) if port else 80)
                except MakiError:
                    continue
                except ValueError:
                    continue  # an unparsable host never connects
                self.fail(f"{url!r} was let through")

    def test_dangerous_schemes_and_malformed_urls_are_refused(self):
        for url in ("file:///etc/passwd", "gopher://10.0.0.1/", "ftp://example.com/x", "javascript:alert(1)",
                    "data:text/plain,hi", "http://", "http:///", "http://:80/", "//127.0.0.1/", "", " ", "http://[::1",
                    "http://exa mple.com/", "http://exa\x00mple.com/", "http://" + "a" * 300 + ".com/"):
            with self.subTest(url=url):
                with self.assertRaises(MakiError):
                    validate_url(url)


class TestResolvedNames(unittest.TestCase):
    """A hostname is judged on what it resolves to, every record of it."""

    def _resolve(self, *addresses):
        infos = []
        for address in addresses:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            infos.append((family, socket.SOCK_STREAM, 6, "", (address, 80)))
        return patch("maki.connector.socket.getaddrinfo", return_value=infos)

    def test_a_name_with_one_refused_record_among_public_ones_is_refused(self):
        for bad in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "::1", "2002:7f00:1::1", "::ffff:10.0.0.1", "224.0.0.1"):
            for order in ((bad, "8.8.8.8"), ("8.8.8.8", bad)):
                with self.subTest(bad=bad, order=order), self._resolve(*order):
                    with self.assertRaises(MakiError):
                        _resolve_and_validate("evil.example.com", 80)

    def test_a_name_with_only_public_records_is_accepted(self):
        with self._resolve("8.8.8.8", "2606:4700:4700::1111"):
            self.assertEqual(_resolve_and_validate("good.example.com", 80), ["8.8.8.8", "2606:4700:4700::1111"])

    def test_well_known_internal_names_are_refused_when_they_resolve_locally(self):
        for name in ("localhost", "LOCALHOST", "localhost.", "ip6-localhost", "localhost.localdomain", "metadata.google.internal"):
            with self.subTest(name=name), self._resolve("127.0.0.1"):
                with self.assertRaises(MakiError):
                    _resolve_and_validate(name, 80)


if __name__ == "__main__":
    unittest.main()
