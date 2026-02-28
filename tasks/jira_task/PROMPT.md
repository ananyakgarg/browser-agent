# Jira Ticket Scrape Task

## Instruction
Go to each Jira ticket URL, take a screenshot, and give me a CSV with the ticket number, assignee, and due date.

## Input CSV
tickets.csv — 5 Jira tickets from https://ananyakgarg25.atlassian.net

## Tickets
- SCRUM-5
- SCRUM-6
- SCRUM-7
- SCRUM-8
- SCRUM-9

## Auth
Interactive login via login.py (Google SSO through real Chrome with remote debugging port, session saved via CDP).

## Results
5/5 succeeded. ~84.6K tokens, 26 API calls, 108 seconds.
