---
id: DE-ENG-320
family: DE-ENG-320
revision: 1
date: 2024-11-14
status: current
lang: en
title: Feeder protection relay baseline settings
kind: reference
---
# Feeder protection relay baseline settings

## 1. Scope
Baseline settings for overcurrent and earth fault relays on 10 kV and 20 kV feeders.

## 2. Overcurrent
The pickup is set to 1.2 times the rated feeder current. The definite time stage trips at 10 times the rated current after 0.1 seconds. Time multiplier setting is 0.3 for normal inverse curves.

## 3. Earth fault
The earth fault pickup is 20 percent of the current transformer primary current. The delay is 0.5 seconds on resistance earthed networks.

## 4. Auto-reclose
Overhead feeders use two auto-reclose attempts. The first dead time is 0.5 seconds and the second is 15 seconds. After the third trip the breaker locks out and the control room is alarmed. Cable feeders are never auto-reclosed.

