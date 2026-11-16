---
id: DE-OPS-130
family: DE-OPS-130
revision: 1
date: 2025-03-18
status: current
lang: en
title: Smart meter installation guide
kind: guide
---
# Smart meter installation guide

## 1. Scope
This guide describes installing a smart meter at a household or small business supply point.

## 2. Meter models
| Model | Phases | Maximum current | Communication |
|---|---|---|---|
| SM-1P | single | 80 A | GPRS |
| SM-3P | three | 100 A | NB-IoT |
| SM-CT | three, CT operated | by CT ratio | NB-IoT with RF backup |

## 3. Installation steps
1. Check that the service fuse does not exceed 63 A for a single-phase meter.
2. Notify the customer that the supply will be off for up to 20 minutes.
3. Isolate, remove the old meter and record its final reading.
4. Fit the smart meter and torque the terminals to 2.5 Nm.
5. Restore the supply and check the signal strength on the meter display.

## 4. Signal and commissioning
The received signal must be at least minus 105 dBm. If it is weaker, fit an RF mesh hub. After power-up the meter must report to the head-end system within 15 minutes, otherwise it is returned to the depot.

## 5. Customer handover
Explain the display and leave the leaflet. The whole visit takes about 45 minutes. Tell the customer that consumption is read remotely and that estimated bills should stop.

