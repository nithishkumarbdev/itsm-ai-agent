---
title: Email Sync Troubleshooting Guide
category: email
version: 1
document_type: troubleshooting
---

## Symptoms
- New emails do not arrive, or arrive late, on one device but not on others.
- Sent emails stay in the outbox.
- The mail app shows "Disconnected", "Trying to connect" or a sync error.
- Calendar items or contacts are missing or out of date.

## Common Causes
- No internet connection, or the VPN is required but not connected.
- The password was changed and the mail app still uses the old one.
- The mail app needs to be restarted, or is out of date.
- The mailbox is full.
- A temporary service problem affecting many users.

## Troubleshooting Steps
1. Confirm the device has internet access. Check whether the mailbox works in the web version of email. If it does, the mailbox is fine and the problem is in the app or device.
2. Fully close the mail app and open it again.
3. If the password was changed recently, sign out of the mail account in the app and sign in again with the current password.
4. Check whether the mailbox is full. If so, ask the user to delete or archive old messages, then wait a few minutes for sync to resume.
5. Check that the app is up to date and that the device date, time and time zone are correct.
6. Restart the device.
7. If the service status page lists an email outage, tell the user that the issue is known and record it in the ticket.
8. Record what works (web mail, other devices) and what does not, along with the exact error message.

## Restricted Actions
- Do not delete mailboxes, emails or mail profiles on behalf of the user.
- Do not change mail server, security or retention settings.
- Do not ask for or accept the user's password.

## When to Escalate
Escalate to IT Support if the problem continues after the steps above, or if web mail also fails.

Escalate to Access Management if sign-in is refused for a valid password.

Escalate to Security if the user reports emails they did not send, or rules and forwarding they did not create.
