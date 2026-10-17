---
title: Password Reset Guide
category: account
version: 1
document_type: procedure
---

## Symptoms
- The user has forgotten their password.
- Sign-in fails with an "incorrect password" message.
- The user is told that the password has expired and must be changed.

## Common Causes
- The password was forgotten or mistyped (caps lock, wrong keyboard layout).
- The password expired under the company password policy.
- The password was changed on one device while another device (phone mail app, saved VPN or Wi-Fi profile) still uses the old one.

## Troubleshooting Steps
1. Ask the user to check caps lock and the keyboard layout, then try once more. Do not let the user retry many times: repeated failures can lock the account (see the Account Lockout Guide).
2. Direct the user to the self-service password reset page, linked from the intranet home page. The user verifies their identity with their registered MFA method and chooses a new password.
3. The new password must follow the rules shown on the reset page: at least 12 characters, a mix of character types, and no reuse of recent passwords.
4. After the reset, ask the user to sign out and sign in again on every device, and to update the saved password on their phone and in any saved VPN or Wi-Fi profile.
5. If self-service reset is not available (no registered MFA method, or the page reports an error), create a ticket for Access Management. A service desk agent must complete the approved identity verification before any manual reset.

## Restricted Actions
- Never reset a password without completing identity verification.
- Never ask a user to tell you their password or an MFA code, and never send a password in chat or email.
- Automated assistants must not reset or change passwords. Only the self-service page or an authorised service desk agent may do so.

## When to Escalate
Escalate to Access Management if self-service reset is unavailable or identity cannot be verified.

Escalate to Security if the user did not request a reset, received MFA prompts they did not start, or suspects someone else knows their password (see the Suspected Account Compromise Guide).
