---
title: MFA Troubleshooting Guide
category: account
version: 1
document_type: troubleshooting
---

## Symptoms
- The MFA prompt or code never arrives.
- The MFA code is rejected as incorrect or expired.
- The user has a new phone and cannot approve sign-ins.
- The user is asked for MFA repeatedly, or approves a prompt and is still refused.

## Common Causes
- The phone has no internet connection or has notifications turned off.
- The phone's date and time are wrong. Time-based codes only work when the phone clock is correct.
- The authenticator app was removed, or the phone was replaced or reset.
- The user is using an old code, or approving a prompt for a different sign-in.

## Troubleshooting Steps
1. Check that the phone has an internet connection and that notifications are allowed for the authenticator app.
2. Check that the phone's date and time are set to automatic.
3. Ask the user to wait for the next code, then enter it without delay, and try again.
4. Ask the user to open the authenticator app directly and approve the pending request, rather than waiting for a notification.
5. If the user has more than one registered method (for example, a text message or a phone call), try the alternative method.
6. If the user has a new phone or lost their old one, they need to register a new device. Access Management must verify the user's identity before any MFA method is reset.
7. Record the method the user normally uses, what happens when they try, and any error message.

## Restricted Actions
- Never disable MFA for a user or an account.
- Never accept, ask for, or pass on an MFA code.
- Automated assistants must not reset MFA methods. Only an authorised Access Management agent can do so, after identity verification.

## When to Escalate
Escalate to Access Management if the user has lost or replaced the MFA device, or cannot use any registered method.

Escalate to Security if the user receives MFA prompts they did not start, since this may mean someone else knows their password (see the Suspected Account Compromise Guide).
