@echo off
rem Pan/tilt tracker GUI - needs Python 3.11 (the install that has
rem cv2 + pyserial; plain "python" here is a newer interpreter
rem without them).
cd /d "%~dp0"
py -3.11 tracker_gui.py
if errorlevel 1 pause
