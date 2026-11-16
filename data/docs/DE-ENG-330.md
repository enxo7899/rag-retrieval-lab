---
id: DE-ENG-330
family: DE-ENG-330
revision: 1
date: 2024-12-03
status: current
lang: en
title: SCADA alarm priorities
kind: reference
---
# SCADA alarm priorities

## 1. Purpose
Defines how alarms in the control room are ranked and answered.

## 2. Priority levels
| Priority | Meaning | Acknowledge within |
|---|---|---|
| 1 Critical | Breaker trip, transformer protection operated | 2 minutes |
| 2 Major | Loss of communication to a substation | 10 minutes |
| 3 Minor | Door open, low battery voltage | 1 hour |
| 4 Information | Routine status change | not required |

## 3. Shelving
An operator may shelve an alarm for at most 8 hours with a written reason. Priority 1 alarms cannot be shelved.

## 4. Review
Alarm statistics are reviewed monthly. A point that raises more than 10 alarms a day is investigated.

