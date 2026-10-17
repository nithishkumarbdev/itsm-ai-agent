---
title: VPN Troubleshooting Guide
category: vpn
version: 1
document_type: troubleshooting
---

## Symptoms
- The VPN client shows "Disconnected" or keeps retrying to connect.
- The connection succeeds but internal resources (shared drives, intranet sites) cannot be reached.
- The client reports an authentication, certificate or timeout error.
- The connection drops every few minutes.

## Common Causes
- No internet connection, or an unstable one, at home or on a public network.
- An expired password, or an MFA prompt that was not approved in time.
- An outdated VPN client version.
- A home router or public Wi-Fi network that blocks VPN traffic.
- The user's account is locked or disabled (see the Account Lockout Guide).

## Troubleshooting Steps
1. Confirm the device has working internet access by opening any public website. If it does not load, follow the Wi-Fi Troubleshooting Guide first.
2. Check that the VPN client is running and shows the company VPN profile. Note the exact error message.
3. Disconnect, wait 30 seconds, and connect again.
4. Confirm the user can sign in with their normal company credentials and approve the MFA prompt on their registered device. If the MFA prompt does not arrive, follow the MFA Troubleshooting Guide.
5. Fully quit the VPN client and start it again. If that does not help, restart the device.
6. Check that the client is up to date. If the client reports that a newer version is available, install it from the Software Center.
7. Try a different network, for example a mobile hotspot, to rule out a home router or public network blocking VPN traffic.
8. Record the results in the ticket, including the exact error message and the time of the last attempt.

## Restricted Actions
- Do not change VPN gateway, firewall or certificate settings. These are managed by the Network team.
- Do not ask the user to disable antivirus, firewall or other security software.
- Do not reset or unlock the user's account from this guide. Use the Account Lockout Guide and its approved process.

## When to Escalate
Escalate to the Network team if:
- the error mentions a certificate, or the VPN profile is missing from the client;
- the connection still fails on more than one network after the steps above;
- several users report the same VPN problem at the same time.

Escalate to Access Management if the account appears locked or disabled.
