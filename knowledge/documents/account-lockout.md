---
title: Account Lockout Guide
category: account
version: 1
document_type: procedure
---

## Symptoms
- Sign-in fails with a message that the account is locked or has been temporarily disabled.
- The password is correct, but sign-in is refused on several devices or applications.
- Several applications ask for credentials at the same time, followed by a lockout.

## Common Causes
- Too many incorrect password attempts.
- An old password saved on a phone, mail app, mapped drive or VPN profile that keeps retrying automatically.
- A recent password change that has not yet reached every device.
- Someone else trying to sign in to the account (see the Suspected Account Compromise Guide).

## Troubleshooting Steps
1. Confirm the message says the account is locked. A wrong-password message is a different problem: see the Password Reset Guide.
2. Ask the user to stop trying to sign in. Every attempt can extend the lockout.
3. Ask the user to sign out of applications and devices that may hold an old password, such as the phone mail app and saved VPN or Wi-Fi profiles.
4. Most lockouts end automatically after the lockout period set by the company policy. Ask the user to wait for that period, and then sign in once with the correct password.
5. If the account is still locked after the waiting period, or the user cannot wait, create a ticket for Access Management. An authorised agent unlocks the account after verifying the user's identity.
6. After the unlock, ask the user to update the saved passwords on all devices.
7. Record when the lockout started and the devices and applications the user was using.

## Restricted Actions
- Do not unlock an account without identity verification by an authorised agent.
- Do not ask the user for their password.
- Automated assistants must not unlock, disable or delete accounts.
- Do not change the account lockout policy.

## When to Escalate
Escalate to Access Management if the account is still locked after the waiting period, or if the user needs urgent access.

Escalate to Security if the account keeps locking while the user is not signing in, or the user reports sign-in attempts they did not make.
