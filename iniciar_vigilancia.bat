@echo off
REM Cambia E:\DCIM por la letra de tu lector de microSD
cd /d "%~dp0"
python procesar_vuelo.py vigilar E:\DCIM --lotes lotes.geojson
pause
