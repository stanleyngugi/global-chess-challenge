@echo off
REM =============================================================================
REM Global Chess Challenge 2025 - Unified Submission Preparation (Windows)
REM =============================================================================
REM
REM Usage:
REM   prepare_submission.bat all
REM   prepare_submission.bat train --epochs 3
REM   prepare_submission.bat validate --num-games 50
REM   prepare_submission.bat package
REM
REM Stages:
REM   data      - Prepare training data
REM   train     - Run LoRA fine-tuning  
REM   merge     - Merge LoRA adapter
REM   validate  - Run validation tests
REM   package   - Create submission package
REM   all       - Run all stages
REM =============================================================================

setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0.."
set "PYTHON=python"
set "SCRIPT=%PROJECT_ROOT%\scripts\prepare_submission.py"

REM Check if Python is available
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

REM Check if script exists
if not exist "%SCRIPT%" (
    echo ERROR: prepare_submission.py not found
    exit /b 1
)

REM Run the Python script with all arguments
%PYTHON% "%SCRIPT%" %*

exit /b %errorlevel%
