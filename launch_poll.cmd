@echo off
REM Windows Scheduled Task entry point: relaunch the 15-minute Kaggle poller
REM if it is not already running. The loop's lock file prevents duplicates.
set ROOT=C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector
powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "%ROOT%\launch_poll.ps1"
