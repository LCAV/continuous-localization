

Measurements

1. Create WiFi hotspot (in report called Robot, password: robot0101)
https://askubuntu.com/questions/318973/how-do-i-create-a-wifi-hotspot-sharing-wireless-internet-connection-single-adap

2. Run "ip neigh"
find mac address starting b8:27...

3. Connect trough ssh
ssh pi@10.42.0.124
password: raspberry (also sudo password)

4. Run command
~/OctoPrint/venv/bin/octoprint serve

5. Open in browser
http://10.42.0.124:5000/

6. Press connect -> will hear funny noise

7. Press the lightning button to turn on or off the device 
through relay (RTT or other) 


8. Terminal -> shows UART of raspberry
run $$ to see all configration
See website for explanations
github/gnea/grbl/wiki/Grbl-v1.1-Configuration

steps per milimeter, 
max speed,
max acceleration,
step idle delay: if motors stay on all the time or not.
Set to 255 to always keep the motors on. 

9. Gcode:
always structure G91 X<distance tavelled by left> 
Y<distance traveled by right>
Can be generated dyrectly with notebook. 

10. Calibration
steps per milimeter: G91 X500 Y500
rotation: G91 X770 Y-770 
Can calculate the parameter from those two lengths
-> divide by pi to get robot width value. (global variables)

11. Drag n drop or upload file to "Files on server"
12. Select file
13. Press print
