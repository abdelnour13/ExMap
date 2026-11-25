@echo off
rem Runs text2kg.ps1 without having to change the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0text2kg.ps1" %*