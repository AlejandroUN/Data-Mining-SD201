@ECHO off

CALL :testCode 1
CALL :testCode 2
CALL :testCode 3
CALL :testCode 4
CALL :testCode 5

CALL :testCode 7
CALL :testCode 8
CALL :testCode 9

ECHO.


EXIT /B %ERRORLEVEL%

:printDiv
set line=-------------------------------------------------------------------------------------------------------------
ECHO. & ECHO %line% & ECHO.
EXIT /B 0

:testCode
CALL :printDiv
python .\main.py debug %~1
python .\check_results.py debug %~1
python .\main.py eval %~1
python .\check_results.py eval %~1
EXIT /B 0
